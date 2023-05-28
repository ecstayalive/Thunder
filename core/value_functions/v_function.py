from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module

from ..nn import MlpBlock2, ReducingCnnBlock

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
            self.features_extractor = nn.Sequential(
                MlpBlock2(observation_features, 256, **factor_kwargs), nn.ReLU6()
            )
            in_features_of_v_net = 256
        else:
            self.features_extractor = nn.Sequential(
                ReducingCnnBlock(observation_features[0], **factor_kwargs),
                nn.ReLU6(),
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
                in_features_of_v_net = self.features_extractor(
                    testing_input_data
                ).shape[1]

            if len(observation_features) == 2:
                self.additional_obs_mlp = nn.Sequential(
                    MlpBlock2(
                        observation_features[1], in_features_of_v_net, **factor_kwargs
                    ),
                    nn.ReLU6(),
                )
        # v net
        self.calculate_v_mlp = MlpBlock2(in_features_of_v_net, 1, **factor_kwargs)

    def forward(self, observation: Tensor, additional_obs: Tensor = None) -> Tensor:
        obs_features = self.features_extractor(observation)
        if additional_obs is not None:
            obs_features += self.additional_obs_mlp(additional_obs)
        return self.calculate_v_mlp(obs_features)
