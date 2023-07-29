########################################################################
# Get the package path
########################################################################
import os
import sys

package_path = os.path.abspath(os.getcwd())
sys.path.append(package_path)
os.chdir(package_path)

import torch
from thunder import BaseTransition, ImageType, MixType, obs_type_checking
from thunder.memory import ReplayBuffer

# testing the storage function
obs = torch.randn(1, 3)
action = torch.randn(1, 2)
reward = torch.randn(1, 1)
next_obs = torch.randn(1, 3)
done = torch.zeros(1, 1)
replay_buffer = ReplayBuffer(5, (3,), (2,))
data_tuple = (obs, action, reward, next_obs, done)
base_transition = BaseTransition(*data_tuple)
replay_buffer.store(base_transition)
print(base_transition["obs"])

# testing type checking function
print(obs_type_checking(5, int))
print(obs_type_checking(((4, 5, 4), 4), MixType))
print(obs_type_checking((3, 4, 5), ImageType))
