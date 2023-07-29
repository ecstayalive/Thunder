import copy

import torch
import torch.nn as nn


def create_target(source: nn.Module) -> nn.Module:
    target = copy.deepcopy(source)
    for param in target.parameters():
        param.requires_grad = False
    return target


def clone_net(source: nn.Module, requires_grad: bool = True) -> nn.Module:
    net = copy.deepcopy(source)
    for param in net.parameters():
        param.requires_grad = requires_grad
    return net


@torch.inference_mode()
def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.005) -> None:
    for tnp, snp in zip(target.parameters(), source.parameters()):
        tnp.data.mul_(1 - tau).add_(snp.data, alpha=tau)


@torch.inference_mode()
def hard_update(target: nn.Module, source: nn.Module) -> None:
    for tnp, snp in zip(target.parameters(), source.parameters()):
        tnp.data.copy_(snp.data)
