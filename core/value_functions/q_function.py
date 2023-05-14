from typing import Tuple, Union

import torch
from torch import nn, Tensor
from torch.nn import Module

from ..nn import CNNDefaultBlock1, MLPDefaultBlock1

ImageType = Tuple[int, int, int]
MixType = Tuple[ImageType, int]


class QFunction(Module):
    def __init__(
        self,
        observation_features: Union[ImageType, int, MixType],
        action_features: int,
        device=None,
        dtype=None,
    ) -> None:
        super(QFunction, self).__init__()
        factor_kwargs = {"device": device, "dtype": dtype}
        if isinstance(observation_features, int):
            self.features_extractor = MLPDefaultBlock1(
                observation_features, 256, **factor_kwargs
            )
            self.action_mpl = MLPDefaultBlock1(action_features, 256, **factor_kwargs)
            self.calculate_q_mlp = MLPDefaultBlock1(256, 1, **factor_kwargs)
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
                CNNDefaultBlock1(observation_tuple[0], **factor_kwargs),
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
                action_mlp_output_features = self.features_extractor(
                    testing_input_data
                ).shape[1]
            self.action_mlp = MLPDefaultBlock1(
                action_dim, action_mlp_output_features, **factor_kwargs
            )
            self.calculate_q_mlp = MLPDefaultBlock1(
                action_mlp_output_features, 1, **factor_kwargs
            )

    def forward(self, observation: Tensor, generalized_action: Tensor) -> Tensor:
        """
        Args:
            observation: When it is int type, it has no other meanings. However,
                         when it is MixType and ImageType. It only stands for image.
            generalized_action: When observation is MixType, it presents an set of
                                some additional observations and actions
        """
        observation_features_vec = self.features_extractor(observation)
        generalized_action_vec = self.action_mlp(generalized_action)
        return self.calculate_q_mlp(observation_features_vec + generalized_action_vec)
