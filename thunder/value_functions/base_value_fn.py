import torch
import torch.nn as nn


class BaseQ(nn.Module):
    q_net: nn.Module

    def __init__(self):
        self.q_net = None

    def forward(
        self,
        obs: torch.Tensor,
        additional_obs: torch.Tensor = None,
        *,
        action: torch.Tensor
    ) -> torch.Tensor:
        self.q_net(obs)
