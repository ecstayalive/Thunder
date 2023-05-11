"""
Jaco robotic arm's basic control framework

"""
from abc import ABC, abstractmethod

import numpy as np
import pybullet as p
import pybullet_data as pd


class Jaco2Agent(ABC):
    """Jaco robotic arm's basic control framework"""

    def __init__(self):
        """Initialization"""
        self.urdf_root = pd.getDataPath()  # << pybullet自带的urdf文件路径
        self.max_force = 200  # << 机械臂机体最大转动力
        self.max_velocity = 0.1  # << 机械臂转动速度
        self.finger_velocity = 0.1  # << 机械臂抓取器转动速度
        self.finger1_force = 2  # << 机械臂手指1的转动力
        self.finger2_force = 2  # << 机械臂手指2的转动力
        self.finger3_force = 2  # << 机械臂手指3的转动力
        self.finger_tip_force = 2  # << 指尖力
        self.finger_pos = 0  # << 手指当前位置
        self.finger_tip_pos = 0  # << 指尖当前位置

    @abstractmethod
    def reset(self):
        """please implement in subclass"""

    @abstractmethod
    def get_end_effector_state(self):
        """Get the state of the jaco's end-effector"""

        """please implement in subclass"""

    @abstractmethod
    def apply_action(self, action):
        """Apply action"""

        """please implement in subclass"""

    @abstractmethod
    def set_original_position(self):
        """Set the robotic arm back to the original state"""

        """please implement in subclass"""

    def accurateCalculateInverseKinematics(
        self,
        jaco_uid,
        end_effector_index,
        target_pos,
        quaternion=None,
        threshold=0.0005,
        max_iter=50,
    ):
        """A more accurate inverse kinematics method

        Args:
            jaco_uid: jaco机械臂id
            end_effector_index: 末端关节id
            target_pos: 目标位置
            quaternion: 目标姿态
            threshold: 定位精度，一般机器人的位置精度在1mm到5mm之间
            max_iter: 最大迭代次数，为减少程序运行的时间，默认为50

        """
        close_enough = False
        iter = 0
        dist2 = 1e8
        while not close_enough and iter < max_iter:
            if quaternion is not None:
                joint_poses = p.calculateInverseKinematics(
                    jaco_uid, end_effector_index, target_pos, quaternion
                )[:6]
            else:
                joint_poses = p.calculateInverseKinematics(
                    jaco_uid, end_effector_index, target_pos
                )[:6]
            for item in enumerate(self.motor_indices[:6]):
                p.resetJointState(jaco_uid, item[1], joint_poses[item[0]])
            ls = p.getLinkState(jaco_uid, end_effector_index)
            new_pos = ls[4]
            diff = [
                target_pos[0] - new_pos[0],
                target_pos[1] - new_pos[1],
                target_pos[2] - new_pos[2],
            ]
            dist2 = np.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            close_enough = dist2 < threshold
            iter += 1
        # print("Num iter: " + str(iter) + "  threshold: " + str(dist2))
        return joint_poses
