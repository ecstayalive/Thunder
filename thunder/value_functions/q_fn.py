from typing import Tuple

import torch
import torch.nn as nn

from thunder.nn.modules.conv_blocks import ReducingConvBlock
from thunder.nn.modules.mlp_blocks import MlpBlock
from thunder.types import ImageType, MixType, obs_type_checking





class GeneralQ(nn.Module):
    def __init__(
        self,
        observation_features: int | ImageType | MixType,
        action_features: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factor_kwargs = {"device": device, "dtype": dtype}
        if isinstance(observation_features, int):
            self.features_extractor = nn.Sequential(
                MlpBlock(observation_features, 256, **factor_kwargs), nn.ReLU6()
            )
            self.action_mlp = nn.Sequential(
                MlpBlock(action_features, 256, **factor_kwargs), nn.ReLU6()
            )
            in_features_of_q_net = 256
        else:
            # NOTE: there are two different ways to implement the deducing network
            #       One is purely MLP, another is CNN blocks plus MLP.
            #       Here we use the purely MLP, because of limited GPU memory.
            #       Maybe later we will change it.
            if len(observation_features) == 3:
                observation_tuple = observation_features
                action_dim = action_features
            elif len(observation_features) == 2:
                observation_tuple = observation_features[0]
                action_dim = observation_features[1] + action_features
            self.features_extractor = nn.Sequential(
                ReducingConvBlock(observation_tuple[0], **factor_kwargs),
                nn.ReLU6(),
                nn.Flatten(),
            )
            with torch.no_grad():
                testing_input_data = torch.zeros(
                    1,
                    observation_tuple[0],
                    observation_tuple[1],
                    observation_tuple[2],
                    **factor_kwargs
                )
                in_features_of_q_net = self.features_extractor(
                    testing_input_data
                ).shape[1]
            self.action_mlp = nn.Sequential(
                MlpBlock(action_dim, in_features_of_q_net, **factor_kwargs), nn.ReLU6()
            )
        self.calculate_q_mlp = MlpBlock(in_features_of_q_net, 1, **factor_kwargs)

    def forward(
        self, obs: torch.Tensor, generalized_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            observation: When it is int type, it has no other meanings. However,
                         when it is MixType and ImageType. It only stands for image.
            generalized_action: When observation is MixType, it presents an set of
                                some additional observations and actions
        """
        observation_features_vec = self.features_extractor(obs)
        generalized_action_vec = self.action_mlp(generalized_data)
        return self.calculate_q_mlp(observation_features_vec + generalized_action_vec)


class MlpQ(nn.nn.Module):
    def __init__(self):
        ...


class CnnQ(nn.nn.Module):
    def __init__(self):
        ...
