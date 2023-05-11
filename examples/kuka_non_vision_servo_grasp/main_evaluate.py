"""Main file
This file is used to evaluate a model for kuka's non-vision servo
grasp task. If there is no model, this file will train one

"""

import os

########################################################################
# Get the package path
########################################################################
import sys

package_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(package_path)
os.chdir(package_path)
import os
import traceback

from core.algorithms import SAC

########################################################################
# Train a model
########################################################################
from envs import KukaNonVisionServoGraspEnv


if __name__ == "__main__":
    env = KukaNonVisionServoGraspEnv(
        render=True,
        width=256,
        height=256,
        show_image=False,
    )
    model = SAC(env=env)
    try:
        model.evaluate_model()
    except BaseException as e:
        traceback.print_exc()
