import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from thunder.nn.mapping import ACTIVATION_INSTANCE_MAP

__all__ = [
    "ReducingConvBlock",
    "AdaptReducingConvBlock",
    "ResidualBasicBlock",
    "ResidualBottleneckBlock",
]


# NOTE: Reinforcement learning can't use max pool as well as batch norm layer.
# TODO: Develop a new CNN architecture which accepts variant size input image.
class ComplexConvBlock:
    ...


class NormalConvBlock:
    ...


class ReducingConvBlock(nn.Module):
    """Reducing Convolution Block, nine convolution layers.

    The feature of this convolution block is that it has fewer parameters,
    and uses strides parameter quickly reducing the image size

    Args:
        in_channels: means input channels
        out_channels: means output channels

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        activation_fn: str = "relu",
        activate_output: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        activation_fn_instance = ACTIVATION_INSTANCE_MAP[activation_fn]
        layers = [
            nn.Conv2d(
                in_channels,
                32,
                kernel_size=8,
                stride=4,
                padding=0,
                **factory_kwargs,
            ),
            activation_fn_instance(),
            ResidualBasicBlock(32, activation_fn, **factory_kwargs),
            activation_fn_instance(),
            nn.Conv2d(
                32,
                64,
                kernel_size=6,
                stride=2,
                padding=0,
                **factory_kwargs,
            ),
            activation_fn_instance(),
            ResidualBasicBlock(64, activation_fn, **factory_kwargs),
            activation_fn_instance(),
            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=0,
                **factory_kwargs,
            ),
            activation_fn_instance(),
            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=0,
                **factory_kwargs,
            ),
            activation_fn_instance(),
            nn.Conv2d(
                256,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=0,
                **factory_kwargs,
            ),
        ]
        if activate_output:
            layers.append(activation_fn_instance())
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv_block(input)


class AdaptReducingConvBlock(nn.Module):
    """Adaptive Reducing Convolution Block, nine convolution layers

    This convolution block has the same architecture of Convolution
    Block 1, which means that it has fewer parameters, and it can
    use strides parameter to reduce the image size quickly. But it
    uses adaptive parameters, which means the channel of the network
    will change with the input channel atomically.

    Args:
        in_channels:

    """

    def __init__(
        self,
        in_channels: int,
        activation_fn: str = "relu",
        activate_output: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        activation_fn_instance = ACTIVATION_INSTANCE_MAP[activation_fn]
        layers = [
            nn.Conv2d(
                in_channels,
                in_channels * 4,
                kernel_size=8,
                stride=4,
                padding=0,
                **factory_kwargs,
            ),
            activation_fn_instance(),
            ResidualBasicBlock(in_channels * 4, activation_fn, **factory_kwargs),
            activation_fn_instance(),
            nn.Conv2d(
                in_channels * 4,
                in_channels * 8,
                kernel_size=6,
                stride=2,
                padding=0,
                **factory_kwargs,
            ),
            activation_fn_instance(),
            ResidualBasicBlock(in_channels * 8, activation_fn, **factory_kwargs),
            activation_fn_instance(),
            nn.Conv2d(
                in_channels * 8,
                in_channels * 16,
                kernel_size=4,
                stride=2,
                padding=0,
                **factory_kwargs,
            ),
            activation_fn_instance(),
            nn.Conv2d(
                in_channels * 16,
                in_channels * 32,
                kernel_size=4,
                stride=2,
                padding=0,
                **factory_kwargs,
            ),
            activation_fn_instance(),
            nn.Conv2d(
                in_channels * 32,
                in_channels * 32,
                kernel_size=3,
                stride=1,
                padding=0,
                **factory_kwargs,
            ),
        ]
        if activate_output:
            layers.append(activation_fn_instance())
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv_block(input)


class ResidualBasicBlock(nn.Module):
    """Basic block in residual network.
    The basic residual convolution block, :math:`y = x + F(x)`.

    NOTE: For details: https://arxiv.org/abs/1512.03385v1

    NOTE: BatchNorm and MaxPool shouldn't used in RL
            http://www.deeprlhub.com/d/469-batch-norm
    """

    def __init__(
        self,
        in_channels: int,
        activation_fn: str = "relu",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        activation_fn_instance = ACTIVATION_INSTANCE_MAP[activation_fn]
        out_channels = in_channels
        straight_pass_layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                **factory_kwargs,
            ),
            activation_fn_instance(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                **factory_kwargs,
            ),
        ]
        self.straight_pass = nn.Sequential(*straight_pass_layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        straight_output = self.straight_pass(input)
        # sourcery skip: inline-immediately-returned-variable
        output = straight_output + input

        return output


class ResidualBottleneckBlock(nn.Module):
    """Bottleneck block in residual network.
    Another residual convolution block, :math:`y = H(x) + F(x)`,
    where the :math:`H(x)` means use 1x1 kernel to process the image.

    NOTE: For details: https://arxiv.org/abs/1512.03385v1

    NOTE: BatchNorm and MaxPool shouldn't used in RL
            http://www.deeprlhub.com/d/469-batch-norm
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_fn: str = "relu",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        activation_fn_instance = ACTIVATION_INSTANCE_MAP[activation_fn]
        if in_channels == out_channels:
            warnings.warn(
                "The input channel should be different with the output channel."
            )
        straight_pass_layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                **factory_kwargs,
            ),
            activation_fn_instance(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                **factory_kwargs,
            ),
        ]
        self.straight_pass = nn.Sequential(*straight_pass_layers)
        self.short_cut_pass = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            **factory_kwargs,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        straight_output = self.straight_pass(input)
        short_cut_output = self.short_cut_pass(input)
        # sourcery skip: inline-immediately-returned-variable
        output = straight_output + short_cut_output
        return output
