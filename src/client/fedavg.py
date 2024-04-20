from collections import OrderedDict
from copy import deepcopy
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from src.utils.tools import (
    NestedNamespace,
    get_optimal_cuda_device,
    trainable_params,
    evalutate_model,
)
from src.utils.metrics import Metrics
from src.utils.models import DecoupledModel
from data.utils.datasets import BaseDataset


class FedAvgClient:
    def __init__(
        self,
        model: DecoupledModel,
        optimizer_cls: type[torch.optim.Optimizer],
        lr_scheduler_cls: type[torch.optim.lr_scheduler.LRScheduler],
        args: NestedNamespace,
        dataset: BaseDataset,
        data_indices: list,
        device: torch.device | None,
        return_diff: bool,
    ):
        self.client_id: int = None
        self.args = args
        if device is None:
            self.device = get_optimal_cuda_device(use_cuda=self.args.common.use_cuda)
        else:
            self.device = device
        self.dataset = dataset
        self.model = model.to(self.device)

        self.optimizer_cls = optimizer_cls
        self.lr_scheduler_cls = lr_scheduler_cls
        self.optimizer = self.optimizer_cls(params=trainable_params(self.model))
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None

        # [{"train": [...], "val": [...], "test": [...]}, ...]
        self.data_indices = data_indices
        # Please don't bother with the [0], which is only for avoiding raising runtime error by setting Subset(indices=[]) with `DataLoader(shuffle=True)`
        self.trainset = Subset(self.dataset, indices=[0])
        self.valset = Subset(self.dataset, indices=[])
        self.testset = Subset(self.dataset, indices=[])
        self.trainloader = DataLoader(
            self.trainset, batch_size=self.args.common.batch_size, shuffle=True
        )
        self.valloader = DataLoader(self.valset, batch_size=self.args.common.batch_size)
        self.testloader = DataLoader(
            self.testset, batch_size=self.args.common.batch_size
        )
        self.testing = False

        self.local_epoch = self.args.common.local_epoch
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        # Some FL methods need it
        self.personal_params_name: list[str] = []

        self.eval_results = {}

        self.return_diff = return_diff
        self.global_regular_model_params: OrderedDict[str, torch.Tensor]

    def load_data_indices(self):
        """This function is for loading data indices for No.`self.client_id` client."""
        self.trainset.indices = self.data_indices[self.client_id]["train"]
        self.valset.indices = self.data_indices[self.client_id]["val"]
        self.testset.indices = self.data_indices[self.client_id]["test"]

    def train_with_eval(self):
        """Wraps `fit()` with `evaluate()` and collect model evaluation results

        A model evaluation results dict: {
                `before`: {...}
                `after`: {...}
                `message`: "..."
            }
            `before` means pre-local-training.
            `after` means post-local-training
        """
        eval_results = {
            "before": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "after": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
        }
        eval_results["before"] = self.evaluate()
        if self.local_epoch > 0:
            self.fit()
            eval_results["after"] = self.evaluate()

        eval_msg = []
        for split, color, flag, subset in [
            ["train", "yellow", self.args.common.eval_train, self.trainset],
            ["val", "green", self.args.common.eval_val, self.valset],
            ["test", "cyan", self.args.common.eval_test, self.testset],
        ]:
            if len(subset) > 0 and flag:
                eval_msg.append(
                    "client [{}] [{}]({})  loss: {:.4f} -> {:.4f}   accuracy: {:.2f}% -> {:.2f}%".format(
                        self.client_id,
                        color,
                        split,
                        eval_results["before"][split].loss,
                        eval_results["after"][split].loss,
                        eval_results["before"][split].accuracy,
                        eval_results["after"][split].accuracy,
                    )
                )
        eval_results["message"] = eval_msg
        self.eval_results = eval_results

    def set_parameters(self, package: dict[str, Any]):
        self.client_id = package["client_id"]
        self.local_epoch = package["local_epoch"]
        self.load_data_indices()

        if package["optimizer_state"]:
            self.optimizer.load_state_dict(package["optimizer_state"])

        if self.lr_scheduler_cls is not None:
            self.lr_scheduler = self.lr_scheduler_cls(optimizer=self.optimizer)
            if package["lr_scheduler_state"]:
                self.lr_scheduler.load_state_dict(package["lr_scheduler_state"])

        self.model.load_state_dict(package["regular_model_params"], strict=False)
        self.model.load_state_dict(package["personal_model_params"], strict=False)
        if self.return_diff:
            self.global_regular_model_params = OrderedDict(
                (key, param.detach().clone().cpu())
                for param, key in zip(*trainable_params(self.model, requires_name=True))
            )

    def train(self, server_package: dict[str, Any]):
        self.set_parameters(server_package)
        self.train_with_eval()
        client_package = self.package()
        return client_package

    def package(self):
        """Package data that client needs to transmit to the server.
        You can override this function and add more parameters.

        Returns:
            A dict: {
                `weight`: Client weight. Defaults to the size of client training set.
                `regular_model_params`: Client model parameters that will join parameter aggregation.
                `model_params_diff`: The parameter difference between the client trained and the global. `diff = global - trained`.
                `eval_results`: Client model evaluation results.
                `personal_model_params`: Client model parameters that absent to parameter aggregation.
                `optimzier_state`: Client optimizer's state dict.
                `lr_scheduler_state`: Client learning rate scheduler's state dict.
            }
        """
        _, regular_keys = trainable_params(self.model, requires_name=True)
        model_params = self.model.state_dict(keep_vars=True)
        client_package = dict(
            weight=len(self.trainset),
            eval_results=self.eval_results,
            regular_model_params={
                key: model_params[key].detach().clone().cpu() for key in regular_keys
            },
            personal_model_params={
                key: param.detach().clone().cpu()
                for key, param in model_params.items()
                if (not param.requires_grad) or (key in self.personal_params_name)
            },
            optimizer_state=deepcopy(self.optimizer.state_dict()),
            lr_scheduler_state={},
        )
        if self.lr_scheduler is not None:
            client_package["lr_scheduler_state"] = deepcopy(
                self.lr_scheduler.state_dict()
            )
        if self.return_diff:
            client_package["model_params_diff"] = {
                key: param_old - param_new
                for (key, param_new), param_old in zip(
                    client_package["regular_model_params"].items(),
                    self.global_regular_model_params.values(),
                )
            }
            client_package.pop("regular_model_params")
        return client_package

    def fit(self):
        self.model.train()
        self.dataset.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                # When the current batch size is 1, the batchNorm2d modules in the model would raise error.
                # So the latent size 1 data batches are discarded.
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module = None) -> dict[str, Metrics]:
        """Evaluating client model.

        Args:
            model: Used model. Defaults to None, which will fallback to `self.model`.

        Returns:
            A evalution results dict: {
                `train`: results on client training set.
                `val`: results on client validation set.
                `test`: results on client testset.
            }
        """
        target_model = self.model if model is None else model
        target_model.eval()
        self.dataset.eval()
        train_metrics = Metrics()
        val_metrics = Metrics()
        test_metrics = Metrics()
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        if len(self.testset) > 0 and self.args.common.eval_test:
            test_metrics = evalutate_model(
                model=target_model,
                dataloader=self.testloader,
                criterion=criterion,
                device=self.device,
            )

        if len(self.valset) > 0 and self.args.common.eval_val:
            val_metrics = evalutate_model(
                model=target_model,
                dataloader=self.valloader,
                criterion=criterion,
                device=self.device,
            )

        if len(self.trainset) > 0 and self.args.common.eval_train:
            train_metrics = evalutate_model(
                model=target_model,
                dataloader=self.trainloader,
                criterion=criterion,
                device=self.device,
            )
        return {"train": train_metrics, "val": val_metrics, "test": test_metrics}

    def test(self, server_package: dict[str, Any]) -> dict[str, dict[str, Metrics]]:
        """Test client model. If `finetune_epoch > 0`, `finetune()` will be activated.

        Args:
            server_package: Parameter package.

        Returns:
            A model evaluation results dict : {
                `before`: {...}
                `after`: {...}
                `message`: "..."
            }
            `before` means pre-local-training.
            `after` means post-local-training
        """
        self.testing = True
        self.set_parameters(server_package)

        results = {
            "before": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "after": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
        }

        results["before"] = self.evaluate()
        if self.args.common.finetune_epoch > 0:
            frz_params_dict = deepcopy(self.model.state_dict())
            self.finetune()
            results["after"] = self.evaluate()
            self.model.load_state_dict(frz_params_dict)

        self.testing = False
        return results

    def finetune(self):
        """Client model finetuning. This function will only be activated in `test()`"""
        self.model.train()
        self.dataset.train()
        for _ in range(self.args.common.finetune_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
