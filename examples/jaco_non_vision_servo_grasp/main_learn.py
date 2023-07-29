"""Main file
This file is used to train a model for jaco's non-vision servo
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

# from core.model import GPModel
from thunder.algorithms import SAC

########################################################################
# Train a model
########################################################################
from envs import JacoNonVisionServoGraspEnv

if __name__ == "__main__":
    jaco = "j2n6s200"
    env = JacoNonVisionServoGraspEnv(
        jaco_model=jaco,
        render=False,
        width=256,
        height=256,
        show_image=False,
    )
    model = SAC(env=env)
    try:
        model.learn(200000)
    except BaseException as e:
        model.save_model()
        traceback.print_exc()
