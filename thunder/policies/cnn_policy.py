from typing import Tuple

import torch
import torch.nn as nn

from thunder import ImageType

from thunder.nn import MlpBlock, ReducingConvBlock
from .base_policy import GaussianPolicy

__all__ = ["CnnGaussianPolicy", "CnnDeterministicPolicy"]

# NOTE: Could we use dropout layer to achieve a stochastic policy.


class CnnGaussianPolicy(GaussianPolicy):
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

    def __init__(
        self,
        in_features: ImageType,
        out_features: int,
        activation_fn: str | Tuple[str, str] = None,
        action_scale: float = 1.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(action_scale)
        factory_kwargs = {"device": device, "dtype": dtype}
        # set different activation fn for conv block and mlp
        if activation_fn is None:
            conv_activation_fn = "relu"
            mlp_activation_fn = "softsign"
        elif isinstance(activation_fn, str):
            conv_activation_fn = activation_fn
            mlp_activation_fn = activation_fn
        else:
            conv_activation_fn = activation_fn[0]
            mlp_activation_fn = activation_fn[1]
        # construct the neural network
        self.features_extractor = nn.Sequential(
            ReducingConvBlock(
                in_features[0], 256, conv_activation_fn, True, **factory_kwargs
            ),
            nn.Flatten(),
        )
        # confirm the input dim of fc layer
        with torch.no_grad():
            conv_out_features = self.features_extractor(
                torch.zeros(
                    1,
                    in_features[0],
                    in_features[1],
                    in_features[2],
                    **factory_kwargs,
                )
            ).shape[1]

        self.mean_net = MlpBlock(
            (conv_out_features, 256, out_features), mlp_activation_fn, **factory_kwargs
        )
        self.log_std_net = MlpBlock(
            (conv_out_features, 256, out_features), mlp_activation_fn, **factory_kwargs
        )


class CnnDeterministicPolicy(nn.Module):
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

    def __init__(
        self,
        in_features: ImageType,
        out_features: int,
        activation_fn: str | Tuple[str, str] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        # set different activation fn for conv block and mlp
        if activation_fn is None:
            conv_activation_fn = "relu"
            mlp_activation_fn = "softsign"
        elif isinstance(activation_fn, str):
            conv_activation_fn = activation_fn
            mlp_activation_fn = activation_fn
        else:
            conv_activation_fn = activation_fn[0]
            mlp_activation_fn = activation_fn[1]
        self.features_extractor = nn.Sequential(
            ReducingConvBlock(
                in_features[0], 256, conv_activation_fn, True, **factory_kwargs
            ),
            nn.Flatten(),
        )
        # confirm the input dim of fc layer
        with torch.no_grad():
            conv_out_features = self.features_extractor(
                torch.zeros(
                    1,
                    in_features[0],
                    in_features[1],
                    in_features[2],
                    **factory_kwargs,
                )
            ).shape[1]

        self.action_mlp = MlpBlock(
            (conv_out_features, 256, out_features), mlp_activation_fn, **factory_kwargs
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        features = self.features_extractor(input)
        return self.action_mlp(features)
