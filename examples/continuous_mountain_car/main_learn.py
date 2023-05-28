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

# from core.model import GPModel
from core.algorithms import SAC

########################################################################
# Train a model
########################################################################
from envs import ContinuousMountainCarEnv

if __name__ == "__main__":
    env = ContinuousMountainCarEnv(render_mode=None, reward_scale=0.01)
    model = SAC(
        env=env,
        buffer_capacity=10000,
        tau=0.005,
        optimizer="Adam",
    )
    try:
        model.learn(
            4000000,
            sample_batch_size=64,
            reward_scale=1 / 8,
            evaluating_period=999,
            evaluating_times=1,
        )
        model.save_model()
    except BaseException as e:
        model.save_model()
        traceback.print_exc()
