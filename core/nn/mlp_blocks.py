from torch import nn, Tensor
from torch.nn import Module

__all__ = ["MLPDefaultBlock"]


class MLPDefaultBlock(Module):
    def __init__(
        self, in_features: int, out_features: int, device=None, dtype=None
    ) -> None:
        super(MLPDefaultBlock, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        hidden_numbers_of_fc = int(2 * (in_features + out_features) / 3)
        self.mlp_block = nn.Sequential(
            nn.Linear(
                in_features,
                hidden_numbers_of_fc,
                **factory_kwargs,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_numbers_of_fc,
                out_features,
                **factory_kwargs,
            ),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.mlp_block(input)
