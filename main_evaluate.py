"""
@file main.py
@author Bruce Hou
@brief 
@version 0.1
@date 2022-01-14

@copyright Copyright (c) 2022
"""

import pybullet as p
import datetime
import numpy as np
from Envs.JacoGraspEnv.JacoVisionServoGraspEnv import JacoVisionServoGraspEnv
from Envs.JacoGraspEnv.JacoNonVisionServoGraspEnv import JacoNonVisionServoGraspEnv
from Envs.KukaGraspEnv.kukaNonVisionServoGraspEnv import KukaNonVisionServoGraspEnv
from Core.Model import GPModel
import traceback
import os


if __name__ == "__main__":
    jaco = "j2n6s200"
    env = JacoNonVisionServoGraspEnv(
        jacoModel=jaco, render=True, width=128, height=128, showImage=False,
    )
    # env = KukaNonVisionServoGraspEnv(renders=True, removeHeightHack=True, width=128, height=128, showImage=True,)
    obs = env.reset()
    model = GPModel(env=env, total_timesteps=5000000)
    try:
        model.evaluate(record=False)
    except BaseException as e:
        traceback.print_exc()
