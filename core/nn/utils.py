import torch.nn as nn
from torch.nn import Module


def init(model: Module) -> None:
    """Initialize network layers
    Use different ways initialize different kinds of layers.

    Args:
        model: the neural network model
    """
    if isinstance(model, nn.Linear):
        nn.init.xavier_normal_(model.weight)
        nn.init.constant_(model.bias, 0)
    elif isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(model, nn.BatchNorm2d):
        nn.init.constant_(model.weight, 1)
        nn.init.constant_(model.bias, 0)
