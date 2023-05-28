from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module

from ..nn import MlpBlock1, MlpBlock2, ReducingCnnBlock

__all__ = ["FeaturesExtractor", "GaussianStochasticPolicy", "DeterministicPolicy"]

# NOTE: Could we use dropout layer to achieve a stochastic policy?
# TODO: Develop a new architecture which accepts variant size input image.


ImageType = Tuple[int, int, int]
MixType = Tuple[ImageType, int]


class FeaturesExtractor(Module):
    """This class includes basic part of policy, which would accept the
    shape of the input and the shape of the output. Then it will
    confirm the network structure and parameter. And it is worth
    noticing that this policy should be changed with your requirement.
    And based this policy, we develop several different kinds of policies,
    which can help accelerate the research.
    But remember, you should develop policy according the requirement of task.
    """

    def __init__(
        self,
        obs_features: Union[int, ImageType, MixType],
        out_features: int,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        # when the obs is one dimension Tensor
        if isinstance(obs_features, int):
            self.features_extractor1 = nn.Sequential(
                MlpBlock2(obs_features, out_features, **factory_kwargs), nn.ReLU6()
            )
            self.features_extractor2 = MlpBlock2(
                out_features, out_features, **factory_kwargs
            )
        # when the obs is a image plus some additional states(observations)
        else:
            # TODO: use a better cnn architecture net work.
            self.features_extractor1 = nn.Sequential(
                ReducingCnnBlock(obs_features[0], **factory_kwargs),
                nn.ReLU6(),
                nn.Flatten(),
            )
            # confirm the input dim of fc layer
            with torch.no_grad():
                out_image_features = self.features_extractor1(
                    torch.zeros(
                        1,
                        obs_features[0],
                        obs_features[1],
                        obs_features[2],
                        **factory_kwargs,
                    )
                ).shape[1]
            self.features_extractor2 = MlpBlock1(
                out_image_features, out_features, **factory_kwargs
            )
            if len(obs_features) == 2:
                additional_obs_features = obs_features[1]
                self.additional_features_extractor = nn.Sequential(
                    MlpBlock2(
                        additional_obs_features, out_image_features, **factory_kwargs
                    ),
                    nn.ReLU6(),
                )

    def forward(self, obs: Tensor, additional_obs: Tensor = None) -> Tensor:
        features = self.features_extractor1(obs)
        if additional_obs is not None:
            features += self.additional_features_extractor(additional_obs)
        features = self.features_extractor2(features)
        return features


class GaussianStochasticPolicy(Module):
    """
    Args:
        obs_features: the features of observation
        action_features: the features of actions
    NOTE:
        In this network architecture, we use fc layer. And to get the number
        of the linear layer, this policy do a virtual forward calculation to
        make sure the neural numbers. But it seems there is a better CNN
        architecture which supports accepting all kinds of size of the input
        image.
    """

    def __init__(
        self,
        obs_features: Union[int, ImageType, MixType],
        action_features: int,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.features_extractor = FeaturesExtractor(obs_features, 256, **factory_kwargs)

        self.mean_mlp = MlpBlock2(256, action_features, **factory_kwargs)
        self.log_std_mlp = MlpBlock2(256, action_features, **factory_kwargs)

    def forward(self, obs: Tensor, additional_obs: Tensor = None) -> Tensor:
        features = self.features_extractor(obs, additional_obs)
        mean = self.mean_mlp(features)
        log_std = self.log_std_mlp(features)

        return mean, log_std


class DeterministicPolicy(Module):
    """
    The policy would accept the shape of the input and the shape of
    the output. Then it will confirm the network structure and parameter.
    And it is worth noticing that this policy should be changed with
    your requirement.

    Args:
        obs_features: the features of observation
        action_features: the features of actions
    NOTE:
        In this network architecture, we use fc layer. And to get the number
        of the linear layer, this policy do a virtual forward calculation to
        make sure the neural numbers. But it seems there is a better CNN
        architecture which supports accepting all kinds of size of the input
        image.

    """

    def __init__(
        self,
        obs_features: Union[int, ImageType, MixType],
        action_features: int,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.features_extractor = FeaturesExtractor(obs_features, 256, **factory_kwargs)
        self.action_mlp = MlpBlock2(512, action_features, **factory_kwargs)

    def forward(self, obs: Tensor, additional_obs: Tensor = None) -> Tensor:
        features = self.features_extractor(obs, additional_obs)
        return self.action_mlp(features)
