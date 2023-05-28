from torch import nn, Tensor
from torch.nn import Module

from ..utils import init_module_

__all__ = ["MlpBlock1", "MlpBlock2"]


class MlpBlock1(Module):
    """MLP with one hidden layer"""

    def __init__(
        self, in_features: int, out_features: int, device=None, dtype=None
    ) -> None:
        super(MlpBlock1, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        hidden_numbers_of_fc = int(2 * (in_features + out_features) / 3)
        self.mlp_block = nn.Sequential(
            nn.Linear(
                in_features,
                hidden_numbers_of_fc,
                **factory_kwargs,
            ),
            nn.Softsign(),
            nn.Linear(
                hidden_numbers_of_fc,
                out_features,
                **factory_kwargs,
            ),
        )
        self.initialize_parameters()

    def initialize_parameters(self):
        init_module_(self.mlp_block)

    def forward(self, input: Tensor) -> Tensor:
        return self.mlp_block(input)


class MlpBlock2(Module):
    """Mlp with two hidden layers"""

    def __init__(
        self, in_features: int, out_features: int, device=None, dtype=None
    ) -> None:
        super(MlpBlock2, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        hidden_numbers_of_fc1 = int(2 * (in_features + out_features) / 3)
        hidden_numbers_of_fc2 = int(2 * (hidden_numbers_of_fc1 + out_features) / 3)
        self.mlp_block = nn.Sequential(
            nn.Linear(
                in_features,
                hidden_numbers_of_fc1,
                **factory_kwargs,
            ),
            nn.Softsign(),
            nn.Linear(
                hidden_numbers_of_fc1,
                hidden_numbers_of_fc2,
                **factory_kwargs,
            ),
            nn.Softsign(),
            nn.Linear(
                hidden_numbers_of_fc2,
                out_features,
                **factory_kwargs,
            ),
        )
        self.initialize_parameters()

    def initialize_parameters(self):
        init_module_(self.mlp_block)

    def forward(self, input: Tensor) -> Tensor:
        return self.mlp_block(input)
