import warnings

import torch.nn.functional as F

from torch import nn, Tensor
from torch.nn import Module

__all__ = ["CNNDefaultBlock1", "ResidualBasicBlock", "ResidualBottleneckBlock"]


class CNNDefaultBlock1(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = 256,
        device=None,
        dtype=None,
    ) -> None:
        """A default model of Convolution Blocks

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
        super(CNNDefaultBlock1, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.feature_exactor = nn.Sequential(
            nn.Conv2d(
                in_features,
                32,
                kernel_size=8,
                stride=4,
                padding=0,
                **factory_kwargs,
            ),
            nn.ReLU(),
            ResidualBasicBlock(32, **factory_kwargs),
            nn.Conv2d(
                32,
                64,
                kernel_size=6,
                stride=2,
                padding=0,
                **factory_kwargs,
            ),
            nn.ReLU(),
            ResidualBasicBlock(64, **factory_kwargs),
            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=0,
                **factory_kwargs,
            ),
            nn.ReLU(),
            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=0,
                **factory_kwargs,
            ),
            nn.ReLU(),
            nn.Conv2d(
                256,
                out_features,
                kernel_size=3,
                stride=1,
                padding=0,
                **factory_kwargs,
            ),
            nn.ReLU(),
            # nn.Flatten(),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.feature_exactor(input)


class CNNDefaultBlock2(Module):
    def __init__(
        self,
        in_features: int,
        device=None,
        dtype=None,
    ) -> None:
        """A default model of Convolution Blocks

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
        super(CNNDefaultBlock2, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.feature_exactor = nn.Sequential(
            nn.Conv2d(
                in_features,
                in_features * 4,
                kernel_size=8,
                stride=4,
                padding=0,
                **factory_kwargs,
            ),
            nn.ReLU(),
            ResidualBasicBlock(in_features * 4, in_features * 4, **factory_kwargs),
            nn.Conv2d(
                in_features * 4,
                in_features * 8,
                kernel_size=6,
                stride=2,
                padding=0,
                **factory_kwargs,
            ),
            nn.ReLU(),
            ResidualBasicBlock(in_features * 8, in_features * 8, **factory_kwargs),
            nn.Conv2d(
                in_features * 8,
                in_features * 16,
                kernel_size=4,
                stride=2,
                padding=0,
                **factory_kwargs,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_features * 16,
                in_features * 32,
                kernel_size=4,
                stride=2,
                padding=0,
                **factory_kwargs,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_features * 32,
                in_features * 32,
                kernel_size=3,
                stride=1,
                padding=0,
                **factory_kwargs,
            ),
            nn.ReLU(),
            # nn.Flatten(),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.feature_exactor(input)


class ResidualBasicBlock(Module):
    """Basic block in residual network.

    Notice: BatchNorm and MaxPool shouldn't used in RL
            http://www.deeprlhub.com/d/469-batch-norm
    """

    def __init__(self, in_channels: int, device=None, dtype=None) -> None:
        super(ResidualBasicBlock, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        out_channels = in_channels
        self.straight_pass = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                **factory_kwargs,
            ),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                **factory_kwargs,
            ),
        )

    def forward(self, input: Tensor) -> Tensor:
        straight_output = self.straight_pass(input)
        return F.relu(straight_output + input)


class ResidualBottleneckBlock(Module):
    """Bottleneck block in residual network.

    Notice: BatchNorm and MaxPool shouldn't used in RL
            http://www.deeprlhub.com/d/469-batch-norm
    """

    def __init__(
        self, in_channels: int, out_channels: int, device=None, dtype=None
    ) -> None:
        super(ResidualBottleneckBlock, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        if in_channels == out_channels:
            warnings.warn(
                "The input channel should be different with the output channel."
            )
        self.straight_pass = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                **factory_kwargs,
            ),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                **factory_kwargs,
            ),
        )
        self.short_cut_pass = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                **factory_kwargs,
            )
        )

    def forward(self, input: Tensor) -> Tensor:
        straight_output = self.straight_pass(input)
        short_cut_output = self.short_cut_pass(input)
        return F.relu(straight_output + short_cut_output)
