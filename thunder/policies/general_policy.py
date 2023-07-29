import torch
import torch.nn as nn

from thunder.nn.mapping import ACTIVATION_INSTANCE_MAP
from thunder.nn.modules import MlpBlock, ReducingConvBlock
from thunder.types import ImageType, MixType, obs_type_checking
from .base_policy import GaussianPolicy

__all__ = ["_FeaturesExtractor", "GeneralGaussianPolicy", "GeneralDeterministicPolicy"]

# NOTE: Could we use dropout layer to achieve a stochastic policy?
# TODO: Develop a new architecture which accepts variant size input image.


class _FeaturesExtractor(nn.Module):
    """This class includes basic part of policy, which would accept the
    shape of the input and the shape of the output. Then it will
    confirm the network structure and parameter. And it is worth
    noticing that this policy should be changed with your requirement.
    And based this policy, we develop several different kinds of policies,
    which can help accelerate the research.
    But remember, you should develop policy according the requirement of task.
    """

    def __init__(
        self,
        obs_features: int | ImageType | MixType,
        out_features: int = 512,
        activation_fn: str = "relu",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        # when the obs is one dimension torch.Tensor
        if obs_type_checking(obs_features, int):
            self.features_extractor1 = MlpBlock(
                (obs_features, 256), activation_fn, True, **factory_kwargs
            )

            self.features_extractor2 = MlpBlock(
                (256, out_features), activation_fn, True, **factory_kwargs
            )
        # when the obs is a image plus some additional states(observations)
        else:
            # TODO: use a better cnn architecture net work.

            # first deal with the different observation type
            if obs_type_checking(obs_features, MixType):
                additional_obs_features = obs_features[1]
                vision_features = obs_features[0]
            else:
                vision_features = obs_features
            # construct features extractor
            self.features_extractor1 = nn.Sequential(
                ReducingConvBlock(
                    vision_features[0], 256, activation_fn, True, **factory_kwargs
                ),
                nn.Flatten(),
            )
            # confirm the input dim of fc layer
            with torch.no_grad():
                conv_out_features = self.features_extractor1(
                    torch.zeros(
                        1,
                        *vision_features,
                        **factory_kwargs,
                    )
                ).shape[1]
            self.features_extractor2 = MlpBlock(
                (conv_out_features, out_features), activation_fn, True, **factory_kwargs
            )
            if obs_type_checking(obs_features, MixType):
                self.additional_features_extractor = MlpBlock(
                    (additional_obs_features, 256, conv_out_features),
                    activation_fn,
                    True,
                    **factory_kwargs,
                )

    def forward(
        self, obs: torch.Tensor, additional_obs: torch.Tensor = None
    ) -> torch.Tensor:
        features = self.features_extractor1(obs)
        if additional_obs is not None:
            features += self.additional_features_extractor(additional_obs)
        features = self.features_extractor2(features)
        return features


class GeneralGaussianPolicy(GaussianPolicy):
    """
    Args:
        obs_features: the features of observation
        action_features: the features of actions
    NOTE:
        In this network architecture, we use fc layer. And to get the number
        of the linear layer, this policy do a virtual forward calculation to
        make sure the neural numbers. But it seems there is a better CNN
        architecture which supports accepting all kinds of size of the input
        image.
    """

    def __init__(
        self,
        obs_features: int | ImageType | MixType,
        action_features: int,
        activation_fn: str = "relu",
        action_scale: float = 1.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(action_scale)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.features_extractor = _FeaturesExtractor(
            obs_features, 512, activation_fn, **factory_kwargs
        )

        self.mean_net = MlpBlock(
            (512, 256, action_features), activation_fn, **factory_kwargs
        )
        self.log_std_net = MlpBlock(
            (512, 256, action_features), activation_fn, **factory_kwargs
        )


class GeneralDeterministicPolicy(nn.Module):
    """
    The policy would accept the shape of the input and the shape of
    the output. Then it will confirm the network structure and parameter.
    And it is worth noticing that this policy should be changed with
    your requirement.

    Args:
        obs_features: the features of observation
        action_features: the features of actions
    NOTE:
        In this network architecture, we use fc layer. And to get the number
        of the linear layer, this policy do a virtual forward calculation to
        make sure the neural numbers. But it seems there is a better CNN
        architecture which supports accepting all kinds of size of the input
        image.

    """

    def __init__(
        self,
        obs_features: int | ImageType | MixType,
        action_features: int,
        activation_fn: str = "relu",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        activation_fn_instance = ACTIVATION_INSTANCE_MAP["activation_fn"]

        self.features_extractor = nn.Sequential(
            _FeaturesExtractor(obs_features, 512, activation_fn, **factory_kwargs),
            activation_fn_instance(),
        )
        self.action_mlp = MlpBlock(
            (512, 256, action_features), activation_fn, **factory_kwargs
        )

    def forward(
        self, obs: torch.Tensor, additional_obs: torch.Tensor = None
    ) -> torch.Tensor:
        features = self.features_extractor(obs, additional_obs)
        return self.action_mlp(features)
