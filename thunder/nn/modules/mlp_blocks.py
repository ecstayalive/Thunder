import math
from typing import Tuple

import torch.nn as nn
from torch import Tensor

from thunder.nn.mapping import ACTIVATION_INSTANCE_MAP


__all__ = ["MlpBlock"]


class MlpBlock(nn.Module):
    """Multi-Layer Perception Block

    Args:
        arch: the architecture of the mlp block,
              for example, one tuple (256, 126, 10) stands
              that the input neurons are 256, the hidden neurons
              are 126 and the output neurons are 10.

        activation_fn: the type of activation function used in
                       this mlp block

    NOTE: Use orthogonal method to initialize the linear weight.
    NOTE: For details: https://arxiv.org/pdf/1609.07093.pdf and
                       https://arxiv.org/pdf/math-ph/0609050.pdf
    """

    def __init__(
        self,
        arch: Tuple,
        activation_fn: str = "softsign",
        activate_output: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        activation_fn_instance = ACTIVATION_INSTANCE_MAP[activation_fn]
        layers = []
        for in_features, out_features in zip(arch[:-2], arch[1:-1]):
            layers.extend(
                (
                    nn.Linear(in_features, out_features, **factory_kwargs),
                    activation_fn_instance(),
                )
            )
        layers.append(nn.Linear(arch[-2], arch[-1], **factory_kwargs))
        if activate_output:
            layers.append(activation_fn_instance())
        self.mlp_block = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self, gain: float = 2.0) -> None:
        for layer in self.mlp_block:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, math.sqrt(gain))

    def forward(self, input: Tensor) -> Tensor:
        return self.mlp_block(input)
