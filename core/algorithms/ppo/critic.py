from typing import Tuple, Union

import numpy as np

import torch
from torch import Tensor

from ...value_functions import VFunction

ImageType = Tuple[int, int, int]
MixType = Tuple[ImageType, int]


class CriticV(VFunction):
    def __init__(
        self, obs_features: Union[int, ImageType, MixType], device=None, dtype=None
    ) -> None:
        self.factor_kwargs = {"device": device, "dtype": dtype}
        super().__init__(obs_features, **self.factor_kwargs)

    def forward(self, obs: Tensor) -> Tensor:
        return super().forward(obs)

    @torch.inference_mode()
    def calc_value(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.tensor(obs, **self.factor_kwargs)
        obs_tensor = obs_tensor.unsqueeze(0)
        value = super().forward(obs_tensor)
        return value.cpu().numpy()
