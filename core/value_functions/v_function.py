from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module

from ..nn import CNNDefaultBlock1, MLPDefaultBlock

ImageType = Tuple[int, int, int]
MixType = Tuple[ImageType, int]


class VFunction(Module):
    def __init__(
        self,
        observation_features: Union[ImageType, int, MixType],
        device=None,
        dtype=None,
    ) -> None:
        super(VFunction, self).__init__()
        factor_kwargs = {"device": device, "dtype": dtype}
        if isinstance(observation_features, int):
            self.features_extractor = MLPDefaultBlock(
                observation_features, 256, **factor_kwargs
            )
            self.calculate_v_mlp = MLPDefaultBlock(256, 1, **factor_kwargs)
        elif len(observation_features) == 3:
            self.features_extractor = nn.Sequential(
                CNNDefaultBlock1(observation_features[0], **factor_kwargs),
                nn.Flatten(),
            )
            with torch.no_grad():
                testing_input_data = torch.zeros(
                    1,
                    observation_features[0],
                    observation_features[1],
                    observation_features[2],
                    **factor_kwargs
                )
                features_numbers = self.features_extractor(testing_input_data).shape[1]
            self.calculate_v_mlp = MLPDefaultBlock(features_numbers, 1, **factor_kwargs)
        elif len(observation_features) == 2:
            observation_features = observation_features[0]
            additional_obs_dim = observation_features[1]
            self.features_extractor = nn.Sequential(
                CNNDefaultBlock1(observation_features[0], **factor_kwargs),
                nn.Flatten(),
            )
            with torch.no_grad():
                testing_input_data = torch.zeros(
                    1,
                    observation_features[0],
                    observation_features[1],
                    observation_features[2],
                    **factor_kwargs
                )
                features_numbers = self.features_extractor(testing_input_data).shape[1]
            self.additional_obs_mlp = MLPDefaultBlock(
                additional_obs_dim, features_numbers, **factor_kwargs
            )
            self.calculate_v_mlp = MLPDefaultBlock(features_numbers, 1, **factor_kwargs)

    def forward(self, observation: Tensor, additional_obs: Tensor = None) -> Tensor:
        if additional_obs is None:
            obs_vec = self.features_extractor(observation)
            obs_vec = F.relu(obs_vec)
            return self.calculate_v_mlp(obs_vec)
        else:
            obs_vec = self.features_extractor(observation)
            additional_obs_vec = self.additional_obs_mlp(additional_obs)
            return self.calculate_v_mlp(obs_vec + additional_obs_vec)
