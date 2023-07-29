from abc import ABC, abstractmethod

import torch


class Distribution(ABC):
    @abstractmethod
    def sample(self) -> torch.Tensor:
        """Sample from the distribution"""

    @abstractmethod
    def rsample(self) -> torch.Tensor:
        """Sample with re-parameterization tricks"""

    @abstractmethod
    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Calculate the probability of the action"""

    @abstractmethod
    def entropy(self) -> torch.Tensor:
        """Calculate the entropy of the distribution"""
