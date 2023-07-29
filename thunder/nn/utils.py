import math

import torch.nn as nn


def orthogonal_module_(module: nn.Module, gain: float = 2.0) -> None:
    """Initialize the network with orthogonal method

    Args:
        module: the neural network
        gain: the gain of the initial params
            Default: 2.0

    NOTE: Usually it is only used in MLP and RNN
    """
    for layer in module.modules():
        nn.init.orthogonal_(layer.weight, math.sqrt(gain))


def xavier_normal_module_(module: nn.Module, gain: float = 1.0) -> None:
    """Initialize the network with xavier normal distribution"""
    for layer in module.modules():
        nn.init.xavier_normal_(layer.weight)
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.bias, gain)
