import warnings

from torch import nn, Tensor
from torch.nn import Module

from ..utils import init_module_

__all__ = [
    "ReducingCnnBlock",
    "AdaptiveReducingCnnBlock",
    "ResidualBasicBlock",
    "ResidualBottleneckBlock",
]

# NOTE: Reinforcement learning can't use max pool as well as batch norm layer.
# TODO: Develop a new CNN architecture which accepts variant size input image.


class ReducingCnnBlock(Module):
    """Reducing Convolution Block, nine convolution layers.

    The feature of this convolution block is that it has fewer parameters,
    and uses strides parameter quickly reducing the image size

    Args:
        in_features: means input channels
        out_features: means output channels

    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 256,
        device=None,
        dtype=None,
    ) -> None:
        super(ReducingCnnBlock, self).__init__()
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
            nn.ReLU(),
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
            nn.ReLU(),
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
            # nn.ReLU(),
            # nn.Flatten(),
        )
        self.initialize_parameters()

    def initialize_parameters(self):
        init_module_(self.feature_exactor)

    def forward(self, input: Tensor) -> Tensor:
        return self.feature_exactor(input)


class AdaptiveReducingCnnBlock(Module):
    """Adaptive Reducing Convolution Block, nine convolution layers

    This convolution block has the same architecture of Convolution
    Block 1, which means that it has fewer parameters, and it can
    use strides parameter to reduce the image size quickly. But it
    uses adaptive parameters, which means the channel of the network
    will change with the input channel atomically.

    Args:
        in_features:

    """

    def __init__(
        self,
        in_features: int,
        device=None,
        dtype=None,
    ) -> None:
        super(AdaptiveReducingCnnBlock, self).__init__()
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
            nn.ReLU(),
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
            nn.ReLU(),
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
            # nn.ReLU(),
            # nn.Flatten(),
        )
        self.initialize_parameters()

    def initialize_parameters(self):
        init_module_(self.feature_exactor)

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
        self.initialize_parameters()

    def initialize_parameters(self):
        init_module_(self.straight_pass)

    def forward(self, input: Tensor) -> Tensor:
        straight_output = self.straight_pass(input)
        return straight_output + input


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
        self.initialize_parameters()

    def initialize_parameters(self):
        init_module_(self.straight_pass)
        init_module_(self.short_cut_pass)

    def forward(self, input: Tensor) -> Tensor:
        straight_output = self.straight_pass(input)
        short_cut_output = self.short_cut_pass(input)
        return straight_output + short_cut_output
