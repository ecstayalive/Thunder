import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATION_INSTANCE_MAP = {
    "softsign": nn.Softsign,
    "tanh": nn.Tanh,
    "relu6": nn.ReLU6,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
}

ACTIVATION_FN_MAP = {
    "softsign": F.softsign,
    "tanh": F.tanh,
    "relu6": F.relu6,
    "relu": F.relu,
    "sigmoid": F.sigmoid,
}

OPTIMIZER_MAP = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}
