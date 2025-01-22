from eegunirep.models import BaseNet, AttentionBaseNet
from eegunirep.utils.utils import get_debug_info

import torch.nn as nn
import torch
import torch.nn.functional as F

class BaseVICReg(BaseNet):
    def __init__(self, device, config, loglevel, dtype = torch.float32):
        super().__init__(device, dtype, loglevel)

        self.sim_coeff = config['model']['sim_coeff']
        self.std_coeff = config['model']['std_coeff']
        self.cov_coeff = config['model']['cov_coeff']
        
        self.mlp_len_list = config['model']['mlp_len_list']
        self.num_features = self.mlp_len_list[-1]
        self.backbone = AttentionBaseNet(config=config, device=self.device, dtype=self.dtype, logger=self.logger).to(device=self.device, dtype=self.dtype)
        self.embedding = self.backbone.out_enc
        self.projector = self.get_projector().to(device=self.device, dtype=self.dtype)
        self.logger.info(f"Vicreg initialized on {device} !")
        self.logger.debug(f"{self}")

    def forward(self, x, batch_size, y=None):
        self.logger.debug(f"input {get_debug_info(x)}") #b, 19, 30, 768
        rep_x = self.backbone(x, batch_size)

        self.logger.debug(f"backbone {get_debug_info(rep_x)}") #b, 2880
        emb_x = self.projector(rep_x)

        self.logger.debug(f"projector {get_debug_info(emb_x)}") #b, 8192
        if y is not None:
            rep_y = self.backbone(y, batch_size)
            emb_y = self.projector(rep_y)
            return rep_x, rep_y, emb_x, emb_y
        else:
            return rep_x, emb_x

        
        
    def get_loss(self, x, y, batch_size):
        repr_loss = F.mse_loss(x, y)
        # print("repr_loss", repr_loss)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
        # print("std_loss", std_loss)

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)
        # print("cov_loss", cov_loss)
        scaled_repr_loss = self.sim_coeff * repr_loss
        scaled_std_loss = self.std_coeff * std_loss
        scaled_cov_loss = self.cov_coeff * cov_loss
        
        return scaled_repr_loss, scaled_std_loss, scaled_cov_loss


    def get_projector(self):
        mlp_spec = [self.embedding] + self.mlp_len_list
        layers = []
        for i in range(len(mlp_spec) - 2):
            layers.append(nn.Linear(mlp_spec[i],mlp_spec[i + 1]))
            layers.append(nn.BatchNorm1d(mlp_spec[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(mlp_spec[-2], mlp_spec[-1], bias=False))
        return nn.Sequential(*layers)

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
