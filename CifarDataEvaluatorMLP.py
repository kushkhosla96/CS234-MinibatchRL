import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from torchsummary import summary

import argparse

import numpy as np
import matplotlib.pyplot as plt

class CifarDataEvaluatorMLP(nn.Module):
    def __init__(self, use_cuda=False, in_size=64*8*8, num_hidden_layers=5):
        super(CifarDataEvaluatorMLP, self).__init__()
        self.in_size = in_size
        self.first_layer = nn.Linear(in_size, in_size // 4)
        self.hiddens = []
        for i in range(1, num_hidden_layers):
            self.hiddens.append(nn.Linear(in_size // 4**i, in_size // 4**(i+1)))
        self.hiddens = nn.ModuleList(self.hiddens)
        self.out_layer = nn.Linear(in_size // 4**(num_hidden_layers), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, self.in_size)
        x = F.relu(self.first_layer(x))
        for hidden in self.hiddens:
            x = F.relu(hidden(x))
        x = self.out_layer(x)
        x = self.sigmoid(x)
        return x
