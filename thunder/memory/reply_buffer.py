from typing import Tuple

import numpy as np

import torch

from thunder import BaseBatch, BaseTransition
from thunder.memory import BaseBuffer

# TODO: Add support for multi-agent simulation environment.


class ReplayBuffer(BaseBuffer):
    """Reply Buffer"""

    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple,
        action_shape: Tuple,
        device: torch.device = None,
        dtype=None,
    ) -> None:
        factor_kwargs = {"device": device, "dtype": dtype}
        super().__init__(capacity, obs_shape, action_shape, **factor_kwargs)
        self.capacity = capacity

    def store(self, t: BaseTransition) -> None:
        """Store one transition data"""

        super().store(t)

    def sample(self, batch_size: int) -> BaseBatch:
        """Uniform sample transitions from buffer

        Args:
            batch_size: The sampling size of transitions

        """
        return super().sample(batch_size)

    def clear(self) -> None:
        """Clear the replay buffer"""
        super().clear()
