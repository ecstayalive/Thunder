"""
J2n6s200's basic control system

"""
import numpy as np
import pybullet as p

from .jaco2_agent import Jaco2Agent


class J2n6s200(Jaco2Agent):
    """Robotic arm control architecture of model J2n6s200"""

    def __init__(
        self,
        base_position=[0, 0, 0],
        jaco_end_effector_position=[0, 0, 0],
        gripper_original_angle=0,
        use_quaternion=True,
        time_step=1.0 / 240,
    ):
        """Initialization

        Args:
            base_position: the initial position of the jaco robotic arm
            jaco_end_effector_position: jaco's end effector position
            gripperAngle: the gripper's close angle
            time_step: one step time for simulation

        """
        super(J2n6s200, self).__init__()

        self.base_position = base_position  # << 默认加载位置
        self.jaco_end_effector_position = jaco_end_effector_position  # << jaco机械臂末端位置
        self.gripper_original_angle = gripper_original_angle  # <<
        self.use_quaternion = use_quaternion  # << 是否使用四元数控制
        self.time_step = time_step  # << 每一步的仿真时间
        if self.use_quaternion:
            # 使用确定位姿，该位姿使得夹爪始终向下
            self.euler = [np.pi, 0, 0]
            self.quaternion = p.getQuaternionFromEuler(self.euler)

        self.jaco_end_effector_index = 7  # << jaco机械臂末端的索引
        self.jaco_finger_motor = [0.45, 0.45]  # << 机械臂抓取器的初始值

        self.reset()

    def reset(self):
        # load the jaco model
        self.jaco_uid = p.loadURDF(
            "urdf/j2n6s200.urdf",
            basePosition=self.base_position,
            useFixedBase=True,
        )
        # acquire the index of jaco motor
        self.num_joints = p.getNumJoints(self.jaco_uid)
        self.motor_names = []
        self.motor_indices = []
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.jaco_uid, i)
            qIndex = joint_info[3]
            if qIndex > -1:
                self.motor_indices.append(i)
                self.motor_names.append(str(joint_info[1]))
        # set the robotic arm's load position
        self.gripper_angle = self.gripper_original_angle  # << jaco机械臂夹爪角度初始值
        self.set_original_position()

    def get_end_effector_state(self):
        """Get end effector height

        Returns: [pos, euler_angle]

        """
        state = p.getLinkState(self.jaco_uid, self.jaco_end_effector_index)
        # 三维坐标
        pos = state[4]
        # 四元数
        qua = state[5]
        euler_angle = p.getEulerFromQuaternion(qua)

        return [pos, euler_angle]

    def apply_action(self, action):
        """The robotic arm performs action

        Args:
            action: [dx, dy, dz, da, finger_angle], finger_angle: close(0) -> open(1)

        """
        # Restrict the pose of the gripper
        self.gripper_angle = self.gripper_angle + action[3]
        if self.gripper_angle >= np.pi:
            self.gripper_angle = np.pi
        if self.gripper_angle <= -np.pi:
            self.gripper_angle = -np.pi

        # Jaco's movement is a process
        dxyz = np.array(action[:3])
        # Restrict the Jaco's workspace
        for i in range(3):
            self.jaco_end_effector_position[i] = (
                self.jaco_end_effector_position[i] + dxyz[i]
            )
        # Restrict x axis
        self.jaco_end_effector_position[0] = min(
            self.jaco_end_effector_position[0], 0.1
        )
        self.jaco_end_effector_position[0] = max(
            self.jaco_end_effector_position[0], -0.1
        )
        # Restrict y axis
        self.jaco_end_effector_position[1] = min(
            self.jaco_end_effector_position[1], 0.2
        )
        self.jaco_end_effector_position[1] = max(
            self.jaco_end_effector_position[1], -0.2
        )
        # Restrict z axis
        self.jaco_end_effector_position[2] = max(
            self.jaco_end_effector_position[2], 0.0
        )
        # Inverse Kinematics
        if self.use_quaternion:
            # 使用确定位姿计算关节角度，但是只能保证夹爪竖直向下
            joint_poses = list(
                # self.accurateCalculateInverseKinematics(
                p.calculateInverseKinematics(
                    self.jaco_uid,
                    self.jaco_end_effector_index,
                    self.jaco_end_effector_position,
                    self.quaternion,
                )
            )
        else:
            # 不使用位姿控制的话直接逆运动学即可
            # joint_poses = self.accurateCalculateInverseKinematics(
            joint_poses = p.calculateInverseKinematics(
                self.jaco_uid,
                self.jaco_end_effector_index,
                self.jaco_end_effector_position,
            )
        # the pose of the end effector, which decide the gripper's angle
        joint_poses[5] = self.gripper_angle
        # 移动机械臂机体
        for item in enumerate(self.motor_indices[:6]):
            p.setJointMotorControl2(
                bodyIndex=self.jaco_uid,
                jointIndex=item[1],
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_poses[item[0]],
                targetVelocity=0,
                force=self.max_force,
                maxVelocity=self.max_velocity,
                positionGain=0.3,
                velocityGain=1,
            )
        # Fingers' movement
        # action[4]: close(0) -> open(1)
        finger_angle = -1.7 * action[4] + 2
        for i in [8, 10]:
            p.setJointMotorControl2(
                self.jaco_uid,
                i,
                p.POSITION_CONTROL,
                targetPosition=finger_angle,
                force=self.finger1_force,
            )
            if action[len(action) - 1] == 1:
                p.setJointMotorControl2(
                    self.jaco_uid,
                    i + 1,
                    p.POSITION_CONTROL,
                    targetPosition=0.7,
                    force=self.finger2_force,
                )

    def set_original_position(self):
        """Set the original position of Jaco arm"""
        # 是否使用确定位姿
        if self.use_quaternion:
            # 当位姿确定，机械臂夹爪始终向下
            jaco_original_position = list(
                self.accurateCalculateInverseKinematics(
                    self.jaco_uid,
                    self.jaco_end_effector_index,
                    self.jaco_end_effector_position,
                    self.quaternion,
                )
            )
        else:
            # 当不使用确定位姿时，就直接进行逆运动学解算
            jaco_original_position = self.accurateCalculateInverseKinematics(
                self.jaco_uid,
                self.jaco_end_effector_index,
                self.jaco_end_effector_position,
            )
        # the pose of the end effector, which decide the gripper's angle
        jaco_original_position[5] = self.gripper_angle
        # 机体
        for item in enumerate(self.motor_indices[:6]):
            p.resetJointState(self.jaco_uid, item[1], jaco_original_position[item[0]])
        # finger
        for i in [8, 10]:
            p.resetJointState(self.jaco_uid, i, 0.3)
            p.resetJointState(self.jaco_uid, i + 1, 0.7)
