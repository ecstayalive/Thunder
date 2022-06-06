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


def setActionSpace(jacoModel, useAngleSpace, isDiscrete, removeZAxis):
    if jacoModel[:5] == "j2n6s":
        if jacoModel[5:8] == "200":
            if isDiscrete:
                if removeZAxis:
                    action_space = spaces.Discrete(9)
                else:
                    action_space = spaces.Discrete(11)
            else:
                if useAngleSpace:
                    action_high = np.array(
                        [
                            2 * np.pi,
                            2.613,
                            2.613,
                            2 * np.pi,
                            2 * np.pi,
                            2 * np.pi,
                            1.362,
                            1.362,
                        ]
                    )
                    action_low = np.array(
                        [
                            -2 * np.pi,
                            -2.618,
                            -2.618,
                            -2 * np.pi,
                            -2 * np.pi,
                            -2 * np.pi,
                            0.45,
                            0.45,
                        ]
                    )
                    action_space = spaces.Box(low=action_low, high=action_high)
                else:
                    if removeZAxis:
                        # dx, dy, da
                        action_space = spaces.Box(low=-10, high=10, shape=(3,))
                    else:
                        action_space = spaces.Box(
                            low=-10, high=10, shape=(4,)
                        )  # dx, dy, dz, da
        elif jacoModel[5:8] == "300":
            if isDiscrete:
                if removeZAxis:
                    action_space = spaces.Discrete(11)
                else:
                    action_space = spaces.Discrete(13)
            else:
                if useAngleSpace:
                    action_high = np.array(
                        [
                            2 * np.pi,
                            2.613,
                            2.613,
                            2 * np.pi,
                            2 * np.pi,
                            2 * np.pi,
                            2,
                            2,
                            2,
                        ]
                    )
                    action_low = np.array(
                        [
                            -2 * np.pi,
                            -2.618,
                            -2.618,
                            -2 * np.pi,
                            -2 * np.pi,
                            -2 * np.pi,
                            0.7,
                            0.7,
                            0.7,
                        ]
                    )
                    action_space = spaces.Box(low=action_low, high=action_high)
                else:
                    if removeZAxis:
                        # dx, dy, d_finger_angle1, d_finger_angle2, d_finger_angle3
                        action_space = spaces.Box(low=-10, high=10, shape=(5,))
                    else:
                        action_space = spaces.Box(
                            low=-10, high=10, shape=(6,)
                        )  # dx, dy, dz, d_finger_angle1, d_finger_angle2, d_finger_angle3
    elif jacoModel[:5] == "j2s7s":
        if jacoModel[5:8] == "200":
            pass
        elif jacoModel[5:8] == "300":
            pass

    return action_space
