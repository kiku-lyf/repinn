import time
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.optim import LBFGS, Adam
from tqdm import tqdm
import argparse
import numpy as np
from model_dict import get_model
import math
import scipy.io

import copy  # Added for get_clones

seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)




def get_perturbation(range, size, type='uniform'):
    if type == 'uniform':
        x_ref = np.linspace(-range, range, size)
        y_ref = np.linspace(-range, range, size)  # Added y_ref
        t_ref = np.linspace(-range, range, size)
        x_mesh, y_mesh, t_mesh = np.meshgrid(x_ref, y_ref, t_ref)  # 3D grid
        x_mesh = torch.from_numpy(x_mesh.flatten()).float()
        y_mesh = torch.from_numpy(y_mesh.flatten()).float()
        t_mesh = torch.from_numpy(t_mesh.flatten()).float()
        return torch.stack((x_mesh, y_mesh, t_mesh), dim=-1)  # (size^3, 3)
    else:
        # Random: generate size^3 points
        x_ref = np.random.uniform(-range, range, size**3)
        y_ref = np.random.uniform(-range, range, size**3)
        t_ref = np.random.uniform(-range, range, size**3)
        return torch.stack((torch.from_numpy(x_ref).float(), torch.from_numpy(y_ref).float(), torch.from_numpy(t_ref).float()), dim=-1)

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])




class WaveAct(nn.Module):
    def __init__(self):
        super(WaveAct, self).__init__()
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)


class Differential_perturbation(nn.Module):
    def __init__(self, range, size):
        super(Differential_perturbation, self).__init__()
        self.perturbation = get_perturbation(range=range, size=size)

    def forward(self, src):
        num_pert = self.perturbation.size(0)
        src_exp = src[:, None, :].expand(-1, num_pert, -1)  # (batch, num_pert, 3)
        perturbed = src_exp + self.perturbation[None, :, :]  # Direct add, since shapes match (3 vs 3)
        return perturbed


class Model(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layer, hidden_d_ff=64):
        super(Model, self).__init__()
        self.linear_emb = nn.Sequential(*[
            nn.Linear(in_dim, hidden_dim // 4),
            WaveAct(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        ])
        self.multiregion_mixer = nn.Sequential(*[
            nn.Linear(4, 8),
            WaveAct(),
            nn.Linear(8, 1),
        ])
        layers = []
        for i in range(num_layer):
            if i == 0:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_d_ff))
                layers.append(WaveAct())
            else:
                layers.append(nn.Linear(in_features=hidden_d_ff, out_features=hidden_d_ff))
                layers.append(WaveAct())
        layers.append(nn.Sequential(*[
            nn.Linear(in_features=hidden_d_ff, out_features=hidden_d_ff),
            WaveAct(),
            nn.Linear(in_features=hidden_d_ff, out_features=out_dim),
        ]))
        self.linear_out = nn.Sequential(*layers)
        self.region_perturbation_1 = Differential_perturbation(range=0.01, size=3)
        self.region_perturbation_2 = Differential_perturbation(range=0.05, size=5)
        self.region_perturbation_3 = Differential_perturbation(range=0.09, size=7)

    def forward(self, x, y, t):
        src = torch.cat((x, y, t), dim=-1)
        # Differential perturbation
        src_1 = self.linear_emb(src)
        src_2 = self.linear_emb(self.region_perturbation_1(src)).mean(dim=1)
        src_3 = self.linear_emb(self.region_perturbation_2(src)).mean(dim=1)
        src_4 = self.linear_emb(self.region_perturbation_3(src)).mean(dim=1)
        # Multi-region mixing
        src = torch.stack([src_1, src_2, src_3, src_4], dim=1)
        e_outputs = self.multiregion_mixer(src.permute(0, 2, 1).contiguous())[:, :, 0]
        output = self.linear_out(e_outputs)
        return output


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Fixed deprecation
        m.bias.data.fill_(0.01)


