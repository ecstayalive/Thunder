from typing import Tuple, Union

import torch
import torch.nn.functional as F
from numpy import ndarray
from torch import Tensor
from torch.distributions import Normal

from ...policies import GaussianStochasticPolicy

ImageType = Tuple[int, int, int]
MixType = Tuple[ImageType, int]


class Actor(GaussianStochasticPolicy):
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

    @torch.inference_mode()
    def act(
        self, obs: ndarray, additional_obs=None, evaluation_mode: bool = False
    ) -> ndarray:
        obs_tensor = torch.tensor(obs, **self.factor_kwargs)
        obs_tensor = obs_tensor.unsqueeze(0)
        if additional_obs is not None:
            additional_obs_tensor = torch.tensor(additional_obs, **self.factor_kwargs)
            additional_obs_tensor = additional_obs_tensor.unsqueeze(0)
        with torch.no_grad():
            raw_mean, raw_std = super().forward(obs_tensor, additional_obs)
        mean = torch.tanh(raw_mean) * self.action_scale
        std = F.softplus(raw_std)
        dist = Normal(mean, std)
        sample_action = dist.sample()
        sample_log_prob = dist.log_prob(sample_action)
        mean_action = torch.tanh(raw_mean) * self.action_scale
        if evaluation_mode:
            return mean_action.squeeze(0).cpu().numpy()
        else:
            return (
                sample_action.squeeze(0).cpu().numpy(),
                sample_log_prob.squeeze(0).cpu().numpy(),
            )

    def calc_log_prob(self, obs: Tensor, action: Tensor) -> None:
        raw_mean, raw_std = super().forward(obs)
        mean = self.action_scale * torch.tanh(raw_mean)
        std = F.softplus(raw_std)
        dist = Normal(mean, std)
        return torch.sum(dist.log_prob(action), dim=-1).unsqueeze(-1)
