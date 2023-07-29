from dataclasses import dataclass
from typing import Tuple

import torch

ImageType = Tuple[int, int, int]

MixType = Tuple[Tuple[int, int, int], int]


def obs_type_checking(obs: int | ImageType | MixType, obs_type: type):
    """Check wether the given obs is belong to its right type

    There are three types about the observation data. One is integral,
    which means the observation data is one dimensional. And the second
    type of the observation data is a image, which is presented by ImageType
    in ```thunder```. The third type of the data is kind of complexity, the
    data is composed by one image and some one dimensional observation data.
    For example, sometimes the observation data are the image that one robot
    sees adding some extra locomotion data, such as: the velocity of the robot.
    In ```thunder```, we use MixType presents this kind situation.

    Args:
        obs: the observation data
        obs_type: the type of the observation data

    NOTE: We have to say that the function may be slow, and it should be used
          as less as possible. This function is usually used in constructing
          a general network.

    """
    # sourcery skip: remove-unnecessary-else
    if isinstance(obs, int):
        return obs_type is int
    else:
        if obs_type is int:
            return False
        if len(obs) != len(obs_type.__args__):
            return False
        is_type_right = True
        for obs_item, obs_type_item in zip(obs, obs_type.__args__):
            if hasattr(obs_type_item, "__args__"):
                is_type_right &= obs_type_checking(obs_item, obs_type_item)
            else:
                is_type_right &= isinstance(obs_item, obs_type_item)
        return is_type_right


@dataclass(slots=True)
class BaseTransition:
    """Basic transition data
    For this class, it only store one transitions,
    and at the same time, the transition must be torch.Tensor type.
    """

    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor

    def keys(self) -> Tuple[str]:
        """Get the value names in transition"""
        return self.__slots__


@dataclass(slots=True)
class BaseBatch:
    """Basic sample transitions data

    This class should be used to sample transitions data
    for training.
    """

    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor

    def keys(self) -> Tuple[list]:
        return self.__slots__
