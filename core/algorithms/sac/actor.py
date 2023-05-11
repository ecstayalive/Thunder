from typing import Tuple

import torch
from numpy import ndarray
from torch import Tensor
from torch.distributions import Normal
from torch.nn import Module

from ...policies import CNNStochasticPolicy

ImageType = Tuple[int, int, int]


class Actor(Module):
    def __init__(
        self, obs_features: ImageType, action_dim: int, device=None, dtype=None
    ) -> None:
        super(Actor, self).__init__()
        self.factor_kwargs = {"device": device, "dtype": dtype}
        self.actor = CNNStochasticPolicy(obs_features, action_dim, **self.factor_kwargs)

    def forward(self):
        raise NotImplementedError("You are not supposed to use this method.")

    def sample(
        self, obs: Tensor, sample_shape: torch.Size() = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        mean, log_std = self.actor(obs)
        std = log_std.exp()
        # A new Gaussian Dist
        action_dist = Normal(mean, std)
        if sample_shape is None:
            sample_action = action_dist.rsample()
        else:
            sample_action = action_dist.rsample(sample_shape)
        # Set action bouncing
        real_sample_action = torch.tanh(sample_action)
        sample_action_log_prob = torch.sum(
            action_dist.log_prob(sample_action), dim=-1
        ).unsqueeze(-1)
        # And also add a correct item to action log probability
        correct_item = torch.sum(
            torch.log(1 - torch.pow(real_sample_action, 2) + 1e-7), dim=-1
        ).unsqueeze(-1)
        action_log_prob = sample_action_log_prob - correct_item
        # # expand the dimension of action entropy
        # action_entropy = -torch.mul(action_prob, action_log_prob).unsqueeze(-1)
        real_mean_action = torch.tanh(mean)
        return real_mean_action, real_sample_action, action_log_prob

    def act(self, obs: ndarray, evaluation_mode: bool = False) -> ndarray:
        obs_tensor = torch.tensor(obs, **self.factor_kwargs)
        obs_tensor = obs_tensor.unsqueeze(0)
        with torch.no_grad():
            if evaluation_mode:
                action, _, _ = self.sample(obs_tensor)
            else:
                _, action, _ = self.sample(obs_tensor)
        return action.squeeze().cpu().numpy()
