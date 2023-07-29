import math
from typing import Any, Callable, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

# TODO: Using thunder distribution package replaces torch distributions


class GaussianPolicy(nn.Module):
    """Base Gaussian policy
    All Gaussian policy should inherit this class. The all you
    need are re-writing the attributions: ```features_extractor```,
    ```mean_net``` and ```log_std_net```, then you could use all
    methods to achieve a stochastic gaussian actor.

    And also, you can re-write ```forward``` method to implement
    your own network forward calculation.
    """

    # TODO: Re-factor this module. The new one should ask the policy
    # TODO: which inherits it re-writing the forward function,
    # TODO: while not ask it re-writing the three attributions.
    # TODO: This can improve the diversity and stability of the policy network.

    features_extractor: nn.Module
    mean_net: nn.Module
    log_std_net: nn.Module
    action_scale: float

    def __init__(self, action_scale: float = 1.0):
        super().__init__()
        self.action_scale = action_scale

    forward: Callable[..., Any]

    def forward(
        self, obs: torch.Tensor, additional_obs: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if additional_obs is None:
            features = self.features_extractor(obs)
        else:
            features = self.features_extractor(obs, additional_obs)
        mean = self.mean_net(features)
        log_std = self.log_std_net(features)

        return mean, log_std

    def rsample(
        self,
        obs: torch.Tensor,
        additional_obs: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Re-parameterization"""
        mean, log_std = self.forward(obs, additional_obs)
        std = log_std.exp()
        distribution = Normal(mean, std)
        sample_data = distribution.rsample()
        # set action bouncing
        sample_action = torch.tanh(sample_data) * self.action_scale
        prob_correction_item = torch.sum(
            torch.log(1 - torch.pow(sample_action, 2) + 1e-7), dim=-1
        )
        action_log_prob = (
            torch.sum(distribution.log_prob(sample_data), dim=-1) - prob_correction_item
        )

        return sample_action, action_log_prob

    def calc_action_log_prob(
        self, mean: torch.Tensor, log_std: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        # normal distribution
        distribution = Normal(mean, log_std.exp())
        # avoid overflow error
        sign = action / torch.abs(action)
        data = torch.atanh((action / self.action_scale) - sign * 1e-5)

        prob_correction_item = torch.sum(
            torch.log(1 - torch.pow(action, 2) + 1e-7), dim=-1
        )
        # sourcery skip: inline-immediately-returned-variable
        action_log_prob = (
            torch.sum(distribution.log_prob(data), dim=-1) - prob_correction_item
        )

        return action_log_prob

    @torch.inference_mode()
    def explore(
        self, obs: torch.Tensor, additional_obs: torch.Tensor = None
    ) -> torch.Tensor:
        mean, log_std = self.forward(obs, additional_obs)
        std = log_std.exp()
        distribution = Normal(mean, std)
        sample_data = distribution.sample()
        # set action bouncing
        sample_action = torch.tanh(sample_data) * self.action_scale
        prob_correction_item = torch.sum(
            torch.log(1 - torch.pow(sample_action, 2) + 1e-7), dim=-1, keepdim=True
        )
        action_log_prob = (
            torch.sum(distribution.log_prob(sample_data), dim=-1, keepdim=True)
            - prob_correction_item
        )

        return sample_action, action_log_prob

    @torch.inference_mode()
    def act(
        self,
        obs: torch.Tensor,
        additional_obs: torch.Tensor = None,
    ) -> torch.Tensor:
        mean, _ = self.forward(obs, additional_obs)
        # sourcery skip: inline-immediately-returned-variable
        action = torch.tanh(mean) * self.action_scale
        return action
