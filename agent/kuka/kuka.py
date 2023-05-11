import os

import numpy as np
import pybullet as p
import pybullet_data as pd


class Kuka:
    def __init__(
        self,
        time_step=1.0 / 240,
        action_apply_time=100,
    ):
        self.urdf_root_path = pd.getDataPath()
        self.time_step = time_step
        self.action_apply_time = action_apply_time

        self.max_velocity = 0.35
        self.max_force = 200.0
        self.finger_a_force = 2.5
        self.finger_b_force = 2.5
        self.finger_tip_force = 2
        self.use_null_space = 21
        self.use_orientation = 1
        self.kuka_end_effector_index = 6
        self.kuka_gripper_index = 7
        # lower limits for null space
        self.ll = [-0.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        # upper limits for null space
        self.ul = [0.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        # joint ranges for null space
        self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        # restposes for null space
        self.rp = [0, 0, 0, 0.5 * np.pi, 0, -np.pi * 0.5 * 0.66, 0]
        # joint damping coefficents
        self.jd = [
            0.00001,
            0.00001,
            0.00001,
            0.00001,
            0.00001,
            0.00001,
            0.00001,
            0.00001,
            0.00001,
            0.00001,
            0.00001,
            0.00001,
            0.00001,
            0.00001,
        ]
        self.reset()

    def reset(self):
        objects = p.loadSDF(
            os.path.join(self.urdf_root_path, "kuka_iiwa/kuka_with_gripper2.sdf")
        )
        self.kuka_uid = objects[0]
        p.resetBasePositionAndOrientation(
            self.kuka_uid,
            [-0.100000, 0.000000, 0.070000],
            [0.000000, 0.000000, 0.000000, 1.000000],
        )
        self.joint_positions = [
            0.006418,
            0.413184,
            -0.011401,
            -1.589317,
            0.005379,
            1.137684,
            -0.006539,
            0.000048,
            -0.299912,
            0.000000,
            -0.000043,
            0.299960,
            0.000000,
            -0.000200,
        ]
        self.num_joints = p.getNumJoints(self.kuka_uid)
        for joint_index in range(self.num_joints):
            p.resetJointState(
                self.kuka_uid, joint_index, self.joint_positions[joint_index]
            )
            p.setJointMotorControl2(
                self.kuka_uid,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=self.joint_positions[joint_index],
                force=self.max_force,
            )
        self.end_effector_pos = [0.537, 0.0, 0.5]
        self.end_effector_angle = 0
        self.motor_names = []
        self.motor_indices = []

        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.kuka_uid, i)
            q_index = joint_info[3]
            if q_index > -1:
                self.motor_names.append(str(joint_info[1]))
                self.motor_indices.append(i)

    def get_end_effector_state(self):
        """Get end effector height

        Returns: [pos, euler_angle]

        """
        state = p.getLinkState(self.kuka_uid, self.kuka_end_effector_index)
        # 三维坐标
        pos = state[4]
        # 四元数
        qua = state[5]
        euler_angle = p.getEulerFromQuaternion(qua)

        return [pos, euler_angle]

    def apply_action(self, action):
        """The robotic arm performs action"""
        # pose, angle of gripper
        da = action[3]
        self.end_effector_angle = self.end_effector_angle + da
        # Kuka's movement is a process
        dxyz = np.array(action[:3])
        for i in range(3):
            self.end_effector_pos[i] = self.end_effector_pos[i] + dxyz[i]
        # Restrict x, y, z axis
        if self.end_effector_pos[0] > 0.65:
            self.end_effector_pos[0] = 0.65
        if self.end_effector_pos[0] < 0.50:
            self.end_effector_pos[0] = 0.50
        if self.end_effector_pos[1] < -0.17:
            self.end_effector_pos[1] = -0.17
        if self.end_effector_pos[1] > 0.22:
            self.end_effector_pos[1] = 0.22
        pos = self.end_effector_pos
        orn = p.getQuaternionFromEuler([0, -np.pi, 0])  # -np.pi,yaw])
        if self.use_null_space == 1:
            if self.use_orientation == 1:
                # joint_poses = self.accurateCalculateInverseKinematics(
                joint_poses = p.calculateInverseKinematics(
                    self.kuka_uid,
                    self.kuka_end_effector_index,
                    pos,
                    orn,
                    self.ll,
                    self.ul,
                    self.jr,
                    self.rp,
                )
            else:
                # joint_poses = self.accurateCalculateInverseKinematics(
                joint_poses = p.calculateInverseKinematics(
                    self.kuka_uid,
                    self.kuka_end_effector_index,
                    pos,
                    lowerLimits=self.ll,
                    upperLimits=self.ul,
                    jointRanges=self.jr,
                    restPoses=self.rp,
                )
        else:
            if self.use_orientation == 1:
                # joint_poses = self.accurateCalculateInverseKinematics(
                joint_poses = p.calculateInverseKinematics(
                    self.kuka_uid,
                    self.kuka_end_effector_index,
                    pos,
                    orn,
                    jointDamping=self.jd,
                )
            else:
                # joint_poses = self.accurateCalculateInverseKinematics(
                joint_poses = p.calculateInverseKinematics(
                    self.kuka_uid, self.kuka_end_effector_index, pos
                )
        for item in enumerate(self.motor_indices[:7]):
            # print(i)
            p.setJointMotorControl2(
                bodyUniqueId=self.kuka_uid,
                jointIndex=item[1],
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_poses[item[0]],
                force=self.max_force,
                maxVelocity=self.max_velocity,
                positionGain=0.3,
                velocityGain=1,
            )
        # gripper's pose
        p.setJointMotorControl2(
            self.kuka_uid,
            7,
            p.POSITION_CONTROL,
            targetPosition=self.end_effector_angle,
            force=self.max_force,
        )
        # close or open finger
        # action[4]: close(0) -> open(1)
        finger_angle = 0.3 * action[4]
        p.setJointMotorControl2(
            self.kuka_uid,
            8,
            p.POSITION_CONTROL,
            targetPosition=-finger_angle,
            force=self.finger_a_force,
        )
        p.setJointMotorControl2(
            self.kuka_uid,
            11,
            p.POSITION_CONTROL,
            targetPosition=finger_angle,
            force=self.finger_b_force,
        )
        p.setJointMotorControl2(
            self.kuka_uid,
            10,
            p.POSITION_CONTROL,
            targetPosition=0,
            force=self.finger_tip_force,
        )
        p.setJointMotorControl2(
            self.kuka_uid,
            13,
            p.POSITION_CONTROL,
            targetPosition=0,
            force=self.finger_tip_force,
        )

    def accurateCalculateInverseKinematics(
        self,
        kukaUid,
        endEffectorIndex,
        targetPos,
        quaternion=None,
        lowerLimits=None,
        upperLimits=None,
        jointRanges=None,
        restPoses=None,
        jointDamping=None,
        threshold=0.0005,
        maxIter=50,
    ):
        closeEnough = False
        iter = 0
        dist2 = 1e8
        while not closeEnough and iter < maxIter:
            if jointDamping is not None and lowerLimits is not None:
                joint_poses = p.calculateInverseKinematics(
                    kukaUid,
                    endEffectorIndex,
                    targetPos,
                    quaternion,
                    lowerLimits,
                    upperLimits,
                    jointRanges,
                    restPoses,
                    jointDamping,
                )
            elif lowerLimits is not None:
                joint_poses = p.calculateInverseKinematics(
                    kukaUid,
                    endEffectorIndex,
                    targetPos,
                    quaternion,
                    lowerLimits,
                    upperLimits,
                    jointRanges,
                    restPoses,
                )
            elif jointDamping is not None:
                joint_poses = p.calculateInverseKinematics(
                    kukaUid,
                    endEffectorIndex,
                    targetPos,
                    quaternion,
                    jointDamping=jointDamping,
                )
            else:
                joint_poses = p.calculateInverseKinematics(
                    kukaUid,
                    endEffectorIndex,
                    targetPos,
                    quaternion,
                )
            for i in range(endEffectorIndex + 1):
                p.resetJointState(kukaUid, i, joint_poses[i])
            ls = p.getLinkState(kukaUid, endEffectorIndex)
            newPos = ls[4]
            diff = [
                targetPos[0] - newPos[0],
                targetPos[1] - newPos[1],
                targetPos[2] - newPos[2],
            ]
            dist2 = np.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            closeEnough = dist2 < threshold
            iter = iter + 1
        # print ("Num iter: "+str(iter) + "threshold: "+str(dist2))
        return joint_poses
