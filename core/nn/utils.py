import torch.nn as nn
from torch.nn import Module


def init_module_(model: Module) -> None:
    """Initialize network layers
    Use different ways initialize different kinds of layers.

    Args:
        model: the neural network model
    """
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
