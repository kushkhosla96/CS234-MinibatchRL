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

class CifarClassifierMLP(nn.Module):
    def __init__(self, use_cuda=False, in_size=84, hidden_dim=100, num_hidden_layers=5):
        super(CifarClassifierMLP, self).__init__()
        self.first_layer = nn.Linear(in_size, hidden_dim)
        self.hiddens = []
        for _ in range(num_hidden_layers):
            self.hiddens.append(nn.Linear(hidden_dim, hidden_dim))
        self.hiddens = nn.ModuleList(self.hiddens)
        self.out_layer = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.first_layer(x))
        for hidden in self.hiddens:
            x = F.relu(hidden(x))
        x = self.out_layer(x)
        x = self.sigmoid(x)
        return x
