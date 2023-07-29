"""Main file
This file is used to train a model for kuka's vision servo
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

from thunder.algorithms import SAC

########################################################################
# Train a model
########################################################################
from envs import KukaVisionServoGraspEnv

if __name__ == "__main__":
    env = KukaVisionServoGraspEnv(
        render=False,
        width=128,
        height=128,
        show_image=True,
    )
    model = SAC(env=env)
    try:
        model.evaluate_model(100)
    except BaseException as e:
        model.save()
        traceback.print_exc()
