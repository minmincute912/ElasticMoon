from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import torch
import os
from torch._tensor import Tensor
from src.server.fedavg import FedAvgServer
from src.client.elasticmoon import ElasticMoonClient
from src.utils.tools import NestedNamespace, trainable_params
import numpy as np

def get_elasticmoon_args(args_list=None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--sample_ratio", type=float, default=0.3)  # opacue
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--mu", type=float, default=0.95)
    # parser.add_argument("--prox",type=float,default=0.01)
    parser.add_argument("--moontau",type=float,default=0.5)
    parser.add_argument("--moonmu",type=float,default=0.0001)

    return parser.parse_args(args_list)


class ElasticMoonServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "ElasticMoon",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=True,
    ):
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        self.init_trainer(ElasticMoonClient)
        layer_num = len(trainable_params(self.model))
        self.clients_sensitivity = [torch.zeros(layer_num) for _ in self.train_clients]
        self.clients_prev_model_params = {i: {} for i in self.train_clients}
        self.boosted_parameters = []
        self.image_counter = 0


    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["sensitivity"] = self.clients_sensitivity[client_id]
        server_package["prev_model_params"] = self.clients_prev_model_params[client_id]
        return server_package

    def train_one_round(self):
        clients_package = self.trainer.train()

        clients_params_diff = []
        clients_weight = []
        clients_sensitivity = []

        save_directory = '/home/dotuanminh/Desktop/elasticmoon1'



        for cid in self.selected_clients:
            self.clients_sensitivity[cid] = clients_package[cid]["sensitivity"]
            clients_params_diff.append(clients_package[cid]["model_params_diff"])
            clients_weight.append(clients_package[cid]["weight"])
            clients_sensitivity.append(clients_package[cid]["sensitivity"])

            sensitivity_values = clients_package[cid]['sensitivity']
            # print(sensitivity_values, cid, self.current_epoch)
            if self.selected_clients and cid == 0:
#                 import seaborn as sns
#                 import matplotlib.pyplot as plt

#     # Thiết lập kiểu dáng của Seaborn
#                 sns.set(style="whitegrid")

#                 plt.figure(figsize=(30, 20))  # Tăng kích thước hình lên 15x10 inches

#                 sensitivity_values = clients_package[cid]["sensitivity"]
# # Tạo một mảng các chỉ số cho các cột
#                 indices = np.arange(len(sensitivity_values))

#                   # Tăng kích thước hình lên 15x10 inches

# # Vẽ biểu đồ cột
#                 sns.barplot(x=indices, y= np.log10(sensitivity_values+1),hue=indices, palette='viridis')

                

# # Thêm nhãn trục và tiêu đề
#                 plt.xlabel('Layer Index', fontsize=35)
#                 plt.ylabel('Sensitivity values', fontsize=35)
#                 plt.title(' ', fontsize=18)
#                 plt.xticks(fontsize=30)
#                 plt.yticks(fontsize=30)
#                 save_path = os.path.join(save_directory, f'sensitivity_values_image_{self.image_counter}.png')
#                 plt.savefig(save_path, format='png', dpi=300)
# Hiển thị biểu đồ

                save_path_txt = os.path.join(save_directory, f'sensitivity_values_client_0_{self.image_counter}.txt')
                # plt.figure(figsize=(30, 20))  # Tăng kích thước hình lên 15x10 inches

                sensitivity_values = clients_package[cid]["sensitivity"]
                sensitivity_values_str = '\n'.join(map(str, sensitivity_values))

                with open(save_path_txt, 'w') as file:
                    file.write(sensitivity_values_str)

                print(f'Saved sensitivity values of client 0 to {save_path_txt}')


        self.image_counter += 1




        for client_id, package in zip(self.selected_clients, clients_package.values()):
            self.clients_prev_model_params[client_id].update(
                package["regular_model_params"]
            )
            self.clients_prev_model_params[client_id].update(
                package["personal_model_params"]
            )


        self.aggregate(clients_weight, clients_params_diff, clients_sensitivity)
        

    def aggregate(
        self,
        clients_weight: list[int],
        clients_params_diff: list[OrderedDict[str, Tensor]],
        clients_sensitivity: list[torch.Tensor],
    ):

        weights = torch.tensor(clients_weight) / sum(clients_weight)
        stacked_sensitivity = torch.stack(clients_sensitivity, dim=-1)
        aggregated_sensitivity = torch.sum(stacked_sensitivity * weights, dim=-1)
        max_sensitivity = stacked_sensitivity.max(dim=-1)[0]
        zeta = 1 + self.args.elasticmoon.tau - aggregated_sensitivity / max_sensitivity


        # print("zeta: {}".format(zeta))




        # zeta_values = zeta.numpy()
        # for z in zeta_values:
        #     boosted_parameters.append(z) if z > 1 else None
        # print((len(boosted_parameters)/14)*100)

        # print(self.boosted_param(zeta))
        # print("")
        # self.boosted_parameters.append(self.boosted_param(zeta))
        # print(sum(self.boosted_parameters)/self.args.common.global_epoch)
        # # print(self.boosted_parameters)




        clients_params_diff_list = [
            list(delta.values()) for delta in clients_params_diff
        ]
        aggregated_diff = [
            torch.sum(weights * torch.stack(diff, dim=-1), dim=-1)
            for diff in zip(*clients_params_diff_list)
        ]

        for param, coef, diff in zip(
            self.global_model_params.values(), zeta, aggregated_diff
        ):
            param.data -= coef * diff


    def boosted_param(self, zeta):
        boosted_param = 0
        len_each_layer = []
        for x in trainable_params(self.model):
            len_each_layer.append(len(x.view(-1)))
        
        for i in range(len(zeta)):
            if zeta[i] > 1:
                boosted_param += len_each_layer[i]

        return float((boosted_param)/sum(len_each_layer))*100
    
