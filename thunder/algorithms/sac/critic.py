from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from ...value_functions import GeneralQ, GeneralV

ImageType = Tuple[int, int, int]
MixType = Tuple[ImageType, int]


class CriticQ(Module):
    def __init__(
        self,
        obs_features: Union[int, ImageType, MixType],
        action_dim: int,
        device=None,
        dtype=None,
    ) -> None:
        super(CriticQ, self).__init__()
        factor_kwargs = {"device": device, "dtype": dtype}
        self.q = GeneralQ(obs_features, action_dim, **factor_kwargs)

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        return self.q(obs, action)


class CriticV(Module):
    def __init__(
        self, obs_features: Union[int, ImageType, MixType], device=None, dtype=None
    ) -> None:
        super(CriticV, self).__init__()
        factor_kwargs = {"device": device, "dtype": dtype}
        self.v = GeneralV(obs_features, **factor_kwargs)

    def forward(self, obs: Tensor) -> Tensor:
        return self.v(obs)
