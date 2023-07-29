from typing import Tuple, Union

import torch
from numpy import ndarray
from torch import Tensor
from torch.distributions import Normal

from thunder.policies import GeneralGaussianPolicy

ImageType = Tuple[int, int, int]
MixType = Tuple[ImageType, int]


class Actor(GeneralGaussianPolicy):
    def __init__(
        self,
        obs_features: Union[int, ImageType, MixType],
        action_features: int,
        action_scale: float = 1.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(obs_features, action_features, device, dtype)
        self.factor_kwargs = {"device": device, "dtype": dtype}
        assert action_scale > 0, "action_scale should be greater than 0"
        self.action_scale = action_scale

    def forward(self):
        raise NotImplementedError("You are not supposed to use this method.")

    def sample(
        self,
        obs: Tensor,
        additional_obs: Tensor = None,
        sample_shape: torch.Size = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        mean, log_std = super().forward(obs, additional_obs)
        std = log_std.exp()
        # Get mean_action
        mean_action = torch.tanh(mean)
        real_mean_action = mean_action * self.action_scale
        # Build a Gaussian Distribution
        distribution = Normal(mean, std)
        if sample_shape is None:
            sample_data = distribution.rsample()
        else:
            sample_data = distribution.rsample(sample_shape)
        # Set action bouncing
        sample_action = torch.tanh(sample_data)
        real_sample_action = sample_action * self.action_scale
        sample_action_log_prob = torch.sum(
            distribution.log_prob(sample_data), dim=-1
        ).unsqueeze(-1)
        # And also add a correct item to action log probability
        correct_item = torch.sum(
            torch.log(1 - torch.pow(sample_action, 2) + 1e-7), dim=-1
        ).unsqueeze(-1)
        action_log_prob = sample_action_log_prob - correct_item
        return real_mean_action, real_sample_action, action_log_prob

    def act(self, obs: ndarray, evaluation_mode: bool = False) -> ndarray:
        obs_tensor = torch.tensor(obs, **self.factor_kwargs)
        obs_tensor = obs_tensor.unsqueeze(0)
        with torch.no_grad():
            if evaluation_mode:
                action, _, _ = self.sample(obs_tensor)
            else:
                _, action, _ = self.sample(obs_tensor)
        return action.squeeze(0).cpu().numpy()
