"""
@author Bruce Hou
@brief 
@version 0.1
@date 2022-01-14
@email ecstayalive@163.com

@copyright Copyright (c) 2022
"""

import numpy as np
from gym import spaces


def getAction(jacoModel, dv, action, useAngleSpace, isDiscrete, removeZAxis):
    if useAngleSpace:
        return action
    else:
        if jacoModel[5:8] == "200":
            if isDiscrete:
                if removeZAxis:
                    dx = [0, -dv, dv, 0, 0, 0, 0, 0, 0,][action]
                    dy = [0, 0, 0, -dv, dv, 0, 0, 0, 0,][action]
                    dz = -dv
                    dfm1 = [0, 0, 0, 0, 0, -0.1, 0.1, 0, 0,][action]
                    dfm2 = [0, 0, 0, 0, 0, 0, 0, -0.1, 0.1,][action]
                else:
                    dx = [0, -dv, dv, 0, 0, 0, 0, 0, 0, 0, 0,][action]
                    dy = [0, 0, 0, -dv, dv, 0, 0, 0, 0, 0, 0,][action]
                    dz = [0, 0, 0, 0, 0, -dv, dv, 0, 0, 0, 0,][action]
                    dfm1 = [0, 0, 0, 0, 0, 0, 0, -0.1, 0.1, 0, 0,][action]
                    dfm2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, -0.1, 0.1,][action]
            else:
                dx = dv * action[0]
                dy = dv * action[1]
                if removeZAxis:
                    dz = -dv
                    da = action[2]
                else:
                    dz = dv * action[2]
                    da = action[3]
            action = [dx, dy, dz, da, 0]
        elif jacoModel[5:8] == "300":
            if isDiscrete:
                if removeZAxis:
                    dx = [0, -dv, dv, 0, 0, 0, 0, 0, 0, 0, 0][action]
                    dy = [0, 0, 0, -dv, dv, 0, 0, 0, 0, 0, 0][action]
                    dz = -dv
                    dfm1 = [0, 0, 0, 0, 0, -0.1, 0.1, 0, 0, 0, 0][action]
                    dfm2 = [0, 0, 0, 0, 0, 0, 0, -0.1, 0.1, 0, 0][action]
                    dfm3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, -0.1, 0.1][action]
                else:
                    dx = [0, -dv, dv, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][action]
                    dy = [0, 0, 0, -dv, dv, 0, 0, 0, 0, 0, 0, 0, 0][action]
                    dz = [0, 0, 0, 0, 0, -dv, dv, 0, 0, 0, 0, 0, 0][action]
                    dfm1 = [0, 0, 0, 0, 0, 0, 0, -0.1, 0.1, 0, 0, 0, 0][action]
                    dfm2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, -0.1, 0.1, 0, 0][action]
                    dfm3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.1, 0.1][action]
            else:
                dx = dv * action[0]
                dy = dv * action[1]
                if removeZAxis:
                    dz = -dv
                    da = action[2]
                else:
                    dz = dv * action[2]
                    da = action[3]
            action = [dx, dy, dz, da, 0]

        return action
