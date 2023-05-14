from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import Module

from ..nn import CNNDefaultBlock1, MLPDefaultBlock1

__all__ = ["CNNStochasticPolicy", "CNNDeterministicPolicy"]

# NOTE: Could we use dropout layer to achieve a stochastic policy.

ImageType = Tuple[int, int, int]


class CNNStochasticPolicy(Module):
    def __init__(
        self,
        in_features: ImageType,
        out_features: int,
        device=None,
        dtype=None,
    ) -> None:
        """Initialization of the CNN policy

        The policy would accept the shape of the input and the shape of
        the output. Then it will confirm the network structure and parameter.
        And it is worth noticing that this policy should be changed with
        your requirement.

        Args:
            in_features:
            out_features:

        NOTE:
            In this network architecture, we use fc layer. And to get the number
            of the linear layer, this policy do a virtual forward calculation to
            make sure the neural numbers. But it seems there is a better CNN
            architecture which supports accepting all kinds of size of the input
            image.
        TODO: Develop a new CNN architecture which accepts variant size input image.

        """
        super(CNNStochasticPolicy, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.feature_exactor = nn.Sequential(
            CNNDefaultBlock1(in_features[0], **factory_kwargs), nn.Flatten()
        )
        # confirm the input dim of fc layer
        with torch.no_grad():
            in_numbers_of_fc = self.feature_exactor(
                torch.zeros(
                    1,
                    in_features[0],
                    in_features[1],
                    in_features[2],
                    **factory_kwargs,
                )
            ).shape[1]

        self.mean_mlp = MLPDefaultBlock1(
            in_numbers_of_fc, out_features, **factory_kwargs
        )
        self.log_std_mlp = MLPDefaultBlock1(
            in_numbers_of_fc, out_features, **factory_kwargs
        )

    def forward(self, input: Tensor) -> Tensor:
        features = self.feature_exactor(input)
        mean = self.mean_mlp(features)
        log_std = self.log_std_mlp(features)

        return mean, log_std


class CNNDeterministicPolicy(Module):
    def __init__(
        self,
        in_features: Tuple[int, int, int],
        out_features: int,
        device=None,
        dtype=None,
    ) -> None:
        """Initialization of the CNN policy

        The policy would accept the shape of the input and the shape of
        the output. Then it will confirm the network structure and parameter.
        And it is worth noticing that this policy should be changed with
        your requirement.

        Args:
            in_features:
            out_features:

        NOTE:
            In this network architecture, we use fc layer. And to get the number
            of the linear layer, this policy do a virtual forward calculation to
            make sure the neural numbers. But it seems there is a better CNN
            architecture which supports accepting all kinds of size of the input
            image.
            TODO: Develop a new CNN architecture which accepts variant size input image.

        """
        super(CNNDeterministicPolicy, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.feature_exactor = nn.Sequential(
            CNNDefaultBlock1(in_features[0], **factory_kwargs), nn.Flatten()
        )
        # confirm the input dim of fc layer
        with torch.no_grad():
            in_numbers_of_fc = self.feature_exactor(
                torch.zeros(
                    1,
                    in_features[0],
                    in_features[1],
                    in_features[2],
                    **factory_kwargs,
                )
            ).shape[1]

        self.action_mlp = MLPDefaultBlock1(
            in_numbers_of_fc, out_features, **factory_kwargs
        )

    def forward(self, input: Tensor) -> Tensor:
        features = self.feature_exactor(input)
        return self.action_mlp(features)
