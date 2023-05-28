"""Main file
This file is used to train a model for Continuous Mountain Car
task.

"""

import os

########################################################################
# Get the package path
########################################################################
import sys

package_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(package_path)
os.chdir(package_path)
import traceback

from core.algorithms import PPO, SAC

########################################################################
# Train a model
########################################################################
from envs import PendulumEnv

if __name__ == "__main__":
    env = PendulumEnv(render_mode=None)
    model = PPO(
        env=env,
        # buffer_capacity=100000,
        # tau=0.005,
        action_scale=2.0,
        optimizer="Adam",
    )
    try:
        model.learn(
            50000,
            reward_scale=1 / 8,
            reward_bias=8,
            evaluating_period=100,
        )
        # model.save_model()
    except BaseException as e:
        # model.save_model()
        traceback.print_exc()
