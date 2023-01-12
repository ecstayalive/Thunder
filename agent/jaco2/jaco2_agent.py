"""
Jaco robotic arm's basic control framework

"""
from abc import ABC, abstractmethod
import pybullet_data as pd


class Jaco2Agent(ABC):
    """Jaco robotic arm's basic control framework"""

    def __init__(self):
        """Initialization"""
        self.urdf_root = pd.getDataPath()  # << pybullet自带的urdf文件路径
        self.max_force = 200  # << 机械臂机体最大转动力
        self.max_velocity = 0.1  # << 机械臂转动速度
        self.finger_velocity= 0.1  # << 机械臂抓取器转动速度
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
