import torch
import torch.nn as nn

from thunder.nn.modules import MlpBlock
from .base_policy import GaussianPolicy

__all__ = ["MlpGaussianPolicy", "MlpDeterministicPolicy"]


class MlpGaussianPolicy(GaussianPolicy):
    def __init__(
        self,
        obs_features: int,
        out_features: int,
        activation_fn: str = "softsign",
        action_scale: float = 1.0,
        device: torch.device = None,
        dtype=None,
    ) -> None:
        super().__init__(action_scale)
        factor_kwargs = {"device": device, "dtype": dtype}
        self.features_extractor = MlpBlock(
            (obs_features, 256), activation_fn, True, **factor_kwargs
        )
        self.mean_net = MlpBlock(
            (256, 512, out_features), activation_fn, **factor_kwargs
        )
        self.log_std_net = MlpBlock(
            (256, 512, out_features), activation_fn, **factor_kwargs
        )


class MlpDeterministicPolicy(nn.Module):
    def __init__(
        self,
        obs_features: int,
        action_features: int,
        activation_fn: str = "softsign",
        device: torch.device = None,
        dtype=None,
    ) -> None:
        super().__init__()
        factor_kwargs = {"device": device, "dtype": dtype}
        self.mlp = MlpBlock(
            (obs_features, 256, 512, action_features), activation_fn, **factor_kwargs
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # sourcery skip: inline-immediately-returned-variable
        action = self.mlp(obs)
        return action
