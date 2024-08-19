from copy import deepcopy
import random
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from torch.nn.functional import cosine_similarity, relu

from src.client.fedavg import FedAvgClient
from src.utils.tools import trainable_params


class ElasticMoonClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)

        self.prev_model = deepcopy(self.model)
        self.global_model = deepcopy(self.model)

        self.training_steps = 0

        self.layer_num = len(trainable_params(self.model))
        self.sensitivity = torch.zeros(self.layer_num, device=self.device)
        self.sampled_trainset = Subset(self.dataset, indices=[])
        self.sampled_trainloader = DataLoader(
            self.sampled_trainset, self.args.common.batch_size
        )

    def load_data_indices(self):
        train_data_indices = deepcopy(self.data_indices[self.client_id]["train"])
        random.shuffle(train_data_indices)
        sampled_size = int(len(train_data_indices) * self.args.elasticmoon.sample_ratio)
        self.sampled_trainset.indices = train_data_indices[:sampled_size]
        self.trainset.indices = train_data_indices[sampled_size:]
        self.valset.indices = self.data_indices[self.client_id]["val"]
        self.testset.indices = self.data_indices[self.client_id]["test"]

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.sensitivity = package["sensitivity"].to(self.device)
        self.global_model.load_state_dict(self.model.state_dict())
        if package["prev_model_params"]:
            self.prev_model.load_state_dict(package["prev_model_params"])
        else:
            self.prev_model.load_state_dict(self.model.state_dict())

    def train(self, server_package: dict[str, Any]):
        self.set_parameters(server_package)

        self.model.eval()
        for x, y in self.sampled_trainloader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            grads_norm = [
                # 1/2*(torch.norm(layer_grad[0]) ** 2 + torch.norm(layer_grad[0],p=1) ** 2)
                torch.norm(layer_grad[0]) ** 2
                for layer_grad in torch.autograd.grad(
                    loss, trainable_params(self.model)
                )
            ]
            for i in range(len(grads_norm)):
                self.sensitivity[i] = (
                    self.args.elasticmoon.mu * self.sensitivity[i]
                    + (1 - self.args.elasticmoon.mu) * grads_norm[i].abs()
                )
            # for i in range(len(grads_norm)):
            #     # Polyak-style averaging
            #     self.training_steps += 1
            #     self.sensitivity[i] = (self.sensitivity[i] * (self.training_steps - 1) + grads_norm[i].abs()) / self.training_steps

        self.train_with_eval()

        client_package = self.package()

        return client_package

    def package(self):
        client_package = super().package()
        client_package["sensitivity"] = self.sensitivity.cpu().clone()
        return client_package

    def fit(self):
        self.model.train()
        self.dataset.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                z_curr = self.model.get_final_features(x, detach=False)
                z_global = self.global_model.get_final_features(x, detach=True)
                z_prev = self.prev_model.get_final_features(x, detach=True)
                logit = self.model.classifier(relu(z_curr))
                loss_sup = self.criterion(logit, y)
                loss_con = -torch.log(
                    torch.exp(
                        cosine_similarity(z_curr.flatten(1), z_global.flatten(1))
                        / self.args.elasticmoon.moontau
                    )
                    / (
                        torch.exp(
                            cosine_similarity(z_prev.flatten(1), z_curr.flatten(1))
                            / self.args.elasticmoon.moontau
                        )
                        + torch.exp(
                            cosine_similarity(z_curr.flatten(1), z_global.flatten(1))
                            / self.args.elasticmoon.moontau
                        )
                    )
                )

                loss = loss_sup + self.args.elasticmoon.moonmu * torch.mean(loss_con)
                # loss = loss_sup + 0.1 * torch.norm(z_curr - z_global) ** 2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()