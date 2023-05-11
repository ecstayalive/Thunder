from torch import Tensor
from torch.nn import Module

__all__ = ["MLPBasePolicy"]


class MLPBasePolicy(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(MLPBasePolicy, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, input: Tensor) -> Tensor:
        pass
