from typing import Tuple

import torch

from thunder.types import BaseBatch, BaseTransition


class BaseBuffer:
    """Basic buffer implementation"""

    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple,
        action_shape: Tuple,
        device: torch.device = None,
        dtype=None,
    ) -> None:
        factor_kwargs = {"device": device, "dtype": dtype}
        # create data chunk to store transitions
        self.obs = torch.zeros(capacity, *obs_shape, **factor_kwargs)
        self.actions = torch.zeros(capacity, *action_shape, **factor_kwargs)
        self.rewards = torch.zeros(capacity, 1, **factor_kwargs)
        self.next_obs = torch.zeros_like(self.obs, **factor_kwargs)
        self.dones = torch.zeros(capacity, 1, **factor_kwargs)
        # some property of the data chunk
        self.capacity = capacity
        self.size: int = 0
        self.ptr: int = 0

    def store(self, t: BaseTransition) -> None:
        """Store one transition data"""
        self.obs[self.ptr] = t.obs
        self.actions[self.ptr] = t.action
        self.rewards[self.ptr] = t.reward
        self.next_obs[self.ptr] = t.next_obs
        self.dones[self.ptr] = t.done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, batch_size: int) -> BaseBatch:
        """Sample several transitions to train the model.

        Args:
            batch_size: The number of sampled transitions
        """

        indices = torch.randint(self.size, (batch_size,))

        sample_obs = self.obs[indices]
        sample_actions = self.actions[indices]
        sample_rewards = self.rewards[indices]
        sample_next_obs = self.next_obs[indices]
        sample_dones = self.dones[indices]

        # sourcery skip: inline-immediately-returned-variable
        sample_transitions = BaseBatch(
            sample_obs,
            sample_actions,
            sample_rewards,
            sample_next_obs,
            sample_dones,
        )

        return sample_transitions

    def clear(self) -> None:
        self.ptr = 0
        self.size = 0

    def to(self, device:torch.device=None) -> None:
        pass

    @property
    def length(self) -> int:
        return self.size

    def __len__(self):
        return self.size
