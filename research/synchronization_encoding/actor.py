import torch
import torch.nn as nn
from thunder.nn import MlpBlock2


class Actor:
    def __init__(
        self, obs_features: int, action_features: int, device=None, dtype=None
    ):
        factor_kwargs = {"device": device, "dtype": dtype}
        self.actor = MlpBlock2(obs_features, action_features, **factor_kwargs)
