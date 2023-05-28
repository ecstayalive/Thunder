"""Main file
This file is used to train a model for kuka's non-vision servo
grasp task.

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

from core.algorithms import SAC

########################################################################
# Train a model
########################################################################
from envs import KukaNonVisionServoGraspEnv

if __name__ == "__main__":
    env = KukaNonVisionServoGraspEnv(
        render=False,
        width=256,
        height=256,
        show_image=False,
    )
    model = SAC(
        env=env,
        buffer_capacity=2000,
        tau=0.001,
        lr=(1e-4, 1e-4, 1e-4),
        optimizer="Adam",
    )
    try:
        model.learn(
            500000,
            sample_batch_size=32,
            reward_scale=0.1,
            evaluating_period=2000,
            evaluating_times=100,
        )
        model.save_model()
    except BaseException as e:
        model.save_model()
        traceback.print_exc()
