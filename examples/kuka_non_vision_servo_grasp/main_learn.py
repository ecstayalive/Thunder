"""Main file
This file is used to train a model for jaco's non-vision servo
grasp task.

"""

########################################################################
# Get the package path
########################################################################
import sys
import os

package_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(package_path)
os.chdir(package_path)
########################################################################
# Train a model
########################################################################
from envs import KukaNonVisionServoGraspEnv
from core.model import GPModel
import traceback

if __name__ == "__main__":
    env = KukaNonVisionServoGraspEnv(
        render=True,
        width=128,
        height=128,
        show_image=False,
    )
    obs = env.reset()
    model = GPModel(env=env, total_time_steps=5000000)
    model.learn()
    try:
        model.learn()
    except BaseException as e:
        model.save()
        traceback.print_exc()
