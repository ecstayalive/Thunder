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
    obs = env.reset()
    model = SAC(env=env)
    try:
        model.learn(1000000)
        model.save_model()
    except BaseException as e:
        model.save_model()
        traceback.print_exc()
