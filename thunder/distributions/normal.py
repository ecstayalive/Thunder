import math

import torch

from .distribution import Distribution


class Normal(Distribution):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean
        self.std = std

    def rsample(self) -> torch.Tensor:
        eps = torch.normal(
            torch.zeros_like(self.mean),
            torch.ones_like(self.std),
        )

        return self.mean + self.std * eps

    def sample(self) -> torch.Tensor:
        return torch.normal(self.mean, self.std)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        ...

    def entropy(self) -> torch.Tensor:
        ...
