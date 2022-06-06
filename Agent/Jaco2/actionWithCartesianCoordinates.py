"""
@author Bruce Hou
@brief Jaco2Agent
@version 0.1
@date 2022-01-14
@email ecstayalive@163.com

@copyright Copyright (c) 2022
"""
import os, inspect
import pybullet as p
import pybullet_data as pd
import os, inspect
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


class Jaco2Agent:
    """
    @brief Agent Jaco2 robotic arm
    """

    def __init__(
        self,
        jacoModel="j2n6s200",
        basePosition=[0, 0, 0],
        jacoEndEffectorPosition=[0, 0, 0],
        urdfRoot=pd.getDataPath(),
        useAngleSpace=True,
        isBinaryGrippers=False,
        gripperAngle=0,
        useQuaternion=False,
        timeStep=1.0 / 240,
        actionApplyTime=500,
    ):
        """
        @brief 初始化
        @param jacoModel: 使用哪一个jaco模型,可选值有"j2n6s200"、"j2n6s200_col"、"j2n6s300"、"j2n6s300_col"、"j2s7s200"、"j2s7s300"
        @param basePosition: 加载jaco机械臂的初始位置
        @param urdfRoot: pybullet所带模型的位置
        @param useAngleSpace: 是否使用角度空间控制jaco机械臂,默认为True
        @param timeStep: 模拟的时间步长
        """
        self._jacoModel = jacoModel  # << jaco机械臂型号
        self._basePosition = basePosition  # << 默认加载位置
        self._jacoEndEffectorPosition = jacoEndEffectorPosition  # << jaco机械臂末端位置
        self._gripperAngle = gripperAngle  # << jaco机械臂夹爪角度初始值
        self._urdfRoot = urdfRoot  # << pybullet自带的urdf文件路径
        self._useQuaternion = useQuaternion  # << 是否使用四元数控制
        self._timeStep = timeStep  # << 每一步的仿真时间
        self.maxForce = 500  # << 机械臂机体最大转动力
        self.maxVelocity = 0.35  # << 机械臂转动速度
        self.fingerVelocity = 0.3  # << 机械臂抓取器转动速度
        self.finger1Force = 5  # << 机械臂手指1的转动力
        self.finger2Force = 5  # << 机械臂手指2的转动力
        self.finger3Force = 5  # << 机械臂手指3的转动力
        self.fingerTipForce = 3  # << 指尖力
        self._isBinaryGrippers = isBinaryGrippers  # << 是否使用开关控制夹具
        self._useAngleSpace = useAngleSpace  # << 是否使用关节角度空间
        self._actionApplyTime = actionApplyTime  # << 电机仿真时间
        if self._useQuaternion:
            # 使用确定位姿，该位姿使得夹爪始终向下
            self._euler = [np.pi, 0, 0]
            self.quaternion = p.getQuaternionFromEuler(self._euler)
        if self._jacoModel[:5] == "j2n6s":
            self.jacoEndEffectorIndex = 7  # << jaco机械臂末端的索引
            if self._jacoModel[5:8] == "200":
                self.jacoFingerMotor = [0.45, 0.45]  # << 机械臂抓取器的初始值
            elif self._jacoModel[5:8] == "300":
                self.jacoFingerMotor = [0.7, 0.7, 0.7]  # << 机械臂抓取器的初始值
        elif self._jacoModel[:5] == "j2s7s":
            # TODO: 对于7自由度的机械臂,末端的索引还为确定
            # self.jacoEndEffectorIndex = 12  # << jaco机械臂末端的索引
            if self._jacoModel[5:8] == "200":
                self.jacoFingerMotor = [0.45, 0.45]  # << 机械臂抓取器的初始值
            elif self._jacoModel[5:8] == "300":
                self.jacoFingerMotor = [0.7, 0.7, 0.7]  # << 机械臂抓取器的初始值

        self.reset()

    def reset(self):
        # set the robotic arm
        self.jacoUid = p.loadURDF(
            os.path.join(self._urdfRoot, "jaco/" + str(self._jacoModel) + ".urdf"),
            basePosition=self._basePosition,
            useFixedBase=True,
        )
        # 设置jaco机械臂初始位置
        self.setOriginalPosition()
        for _ in range(int(self._actionApplyTime / 4)):
            p.stepSimulation()
        # 获得jaco机械臂的电机关节索引
        self.numJoints = p.getNumJoints(self.jacoUid)
        self.motorNames = []
        self.motorIndices = []
        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.jacoUid, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                self.motorIndices.append(i)
                self.motorNames.append(str(jointInfo[1]))

        # # 就是每次调用电机后，一定得调用一下stepSimulation，否则电机不会转动，获得的信息都是错误的
        # state = p.getLinkState(bodyUniqueId=self.jacoUid, linkIndex=self.jacoEndEffectorIndex, computeForwardKinematics=1)
        # print(state)
        # qua = state[1]
        # euler = p.getEulerFromQuaternion(qua)
        # print(euler)

    def getActionDimension(self):
        """
        @brief 获得运动空间纬度
        """
        pass

    def getObservation(self):
        """
        @brief 获得机械臂内部电机状态
        """
        # TODO： 确定逻辑
        pass

    def applyAction(self, action):
        """
        @brief 执行动作
        """
        # 关节角度空间
        if self._useAngleSpace:
            if self._jacoModel[:5] == "j2n6s":
                # 使用关节角度空间，对于该型号的机械臂而言，信号有两种
                # 一种是机体六个关节角度加夹爪的开关信号
                # 一种是机体六个关节角度加夹爪的多个电机信号
                # 机械臂的6个自由度， 不受是否使用开关信号的控制
                for i in range(1, 7):
                    p.setJointMotorControl2(
                        bodyIndex=self.jacoUid,
                        jointIndex=i,
                        controlMode=p.POSITION_CONTROL,
                        targetVelocity=0,
                        targetPosition=action[i - 1],
                        force=self.maxForce,
                        maxVelocity=self.maxVelocity,
                        positionGain=0.3,
                        velocityGain=1,
                    )
                if self._jacoModel[5:8] == "200":
                    # finger
                    if self._isBinaryGrippers:
                        # 开关信号在最后一位，0或1
                        if action[len(action) - 1] == 1:
                            for i in range(2):
                                p.setJointMotorControl2(
                                    bodyIndex=self.jacoUid,
                                    jointIndex=8 + i * 2,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=1.362,
                                    force=self.finger1Force,
                                )
                        else:
                            for i in range(2):
                                p.setJointMotorControl2(
                                    bodyIndex=self.jacoUid,
                                    jointIndex=8 + i * 2,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=0.45,
                                    force=self.finger1Force,
                                )

                    else:
                        # 手指的两个电机信号
                        for i in range(2):
                            if action[i + 6] > 1.362:
                                action[i + 6] = 1.362
                            p.setJointMotorControl2(
                                bodyIndex=self.jacoUid,
                                jointIndex=8 + i * 2,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=action[6 + i],
                                force=self.finger1Force,
                            )
                elif self._jacoModel[5:8] == "300":
                    # finger
                    if self._isBinaryGrippers:
                        # 开关信号在最后一位，0或1
                        if action[len(action) - 1] == 1:
                            for i in range(2):
                                p.setJointMotorControl2(
                                    bodyIndex=self.jacoUid,
                                    jointIndex=8 + i * 2,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=2,
                                    force=self.finger1Force,
                                )
                        else:
                            for i in range(2):
                                p.setJointMotorControl2(
                                    bodyIndex=self.jacoUid,
                                    jointIndex=8 + i * 2,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=0.7,
                                    force=self.finger1Force,
                                )
                    else:
                        # 手指的三个电机信号
                        for i in range(3):
                            if action[i + 6] > 2:
                                action[i + 6] = 2
                            p.setJointMotorControl2(
                                bodyIndex=self.jacoUid,
                                jointIndex=8 + i * 2,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=action[6 + i],
                                force=self.finger1Force,
                            )
            elif self._jacoModel[:5] == "j2s7s":
                pass
        else:
            # TODO: 对于非关节角度空间位姿问题
            # 前三位的信号始终时xyz的增量
            # 如果使用确定位姿，则有5位，倒数第二位是旋转角度，倒数第一位是夹具开关信号
            # 如果不使用确定位姿，则后几位是手指电机信号
            if self._jacoModel[:5] == "j2n6s":
                # 前三位是坐标增量信息，始终不变
                dxyz = action[:3]
                # 限制机械臂工作区间,只限制x y轴
                for i in range(3):
                    self._jacoEndEffectorPosition[i] = (
                        self._jacoEndEffectorPosition[i] + dxyz[i]
                    )
                for i in range(2):
                    if self._jacoEndEffectorPosition[i] > 0.25:
                        self._jacoEndEffectorPosition[i] = 0.25
                    if self._jacoEndEffectorPosition[i] < -0.25:
                        self._jacoEndEffectorPosition[i] = -0.25
                # 限制z轴
                if self._jacoEndEffectorPosition[2] < 0:
                    self._jacoEndEffectorPosition[2] = 0

                if self._isBinaryGrippers and self._useQuaternion:
                    # 当时用开关信号的时侯，需要进行位姿控制，但是机械臂夹爪始终竖直向下
                    # 如果使用开关信号，倒数第二个量便是夹爪的角度增量值
                    # 倒数第一个量是夹爪的开关信号
                    self._gripperAngle = self._gripperAngle + action[len(action) - 2]
                    if self._gripperAngle >= np.pi * 2:
                        self._gripperAngle = np.pi * 2
                    if self._gripperAngle <= 0:
                        self._gripperAngle = 0
                else:
                    # 如果不使用开关信号，则最后几位便是夹爪的电机角度增量值
                    dfinger_motor = action[3 : len(action)]
                    # 限制机械臂手指位置
                    for i in range(len(dfinger_motor)):
                        self.jacoFingerMotor[i] = (
                            self.jacoFingerMotor[i] + dfinger_motor[i]
                        )
                    for i in range(len(dfinger_motor)):
                        if self._jacoModel[5:8] == "200":
                            if self.jacoFingerMotor[i] > 1.362:
                                self.jacoFingerMotor[i] = 1.362
                            if self.jacoFingerMotor[i] < 0.45:
                                self.jacoFingerMotor[i] = 0.45
                        elif self._jacoModel[5:8] == "300":
                            if self.jacoFingerMotor[i] > 2:
                                self.jacoFingerMotor[i] = 2
                            if self.jacoFingerMotor[i] < 0.7:
                                self.jacoFingerMotor[i] = 0.7
                # 逆运动学
                if self._useQuaternion and self._isBinaryGrippers:
                    # 使用确定位姿计算关节角度，但是只能保证夹爪竖直向下
                    jointPoses = list(
                        self.accurateCalculateInverseKinematics(
                            self.jacoUid,
                            self.jacoEndEffectorIndex,
                            self._jacoEndEffectorPosition,
                            self.quaternion,
                        )
                    )
                    # 夹爪竖直向下，位姿控制直接改变最后一个电机值
                    jointPoses[5] = self._gripperAngle

                else:
                    # 不使用位姿控制的话直接逆运动学即可
                    jointPoses = self.accurateCalculateInverseKinematics(
                        self.jacoUid,
                        self.jacoEndEffectorIndex,
                        self._jacoEndEffectorPosition,
                    )
                # 移动机械臂机体
                for i in range(1, 7):
                    p.setJointMotorControl2(
                        bodyIndex=self.jacoUid,
                        jointIndex=i,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=jointPoses[i - 1],
                        targetVelocity=0,
                        force=self.maxForce,
                        maxVelocity=self.maxVelocity,
                        positionGain=0.3,
                        velocityGain=1,
                    )
                if self._jacoModel[5:8] == "200":
                    # fingers
                    if self._isBinaryGrippers:
                        # 最后一位便是开关信号
                        if action[len(action) - 1] == 1:
                            for i in range(2):
                                p.setJointMotorControl2(
                                    self.jacoUid,
                                    8 + i * 2,
                                    p.POSITION_CONTROL,
                                    targetPosition=1.362,
                                    force=self.finger1Force,
                                )
                        else:
                            for i in range(2):
                                p.setJointMotorControl2(
                                    self.jacoUid,
                                    8 + i * 2,
                                    p.POSITION_CONTROL,
                                    targetPosition=0.45,
                                    force=self.finger1Force,
                                )
                    else:
                        # 最后两位时电机角度值
                        for i in range(2):
                            p.setJointMotorControl2(
                                self.jacoUid,
                                8 + i * 2,
                                p.POSITION_CONTROL,
                                targetPosition=self.jacoFingerMotor[i],
                                force=self.finger1Force,
                            )
                elif self._jacoModel[5:8] == "300":
                    # fingers
                    if self._isBinaryGrippers:
                        # 最后一位便是开关信号
                        if action[len(action) - 1] == 1:
                            for i in range(2):
                                p.setJointMotorControl2(
                                    self.jacoUid,
                                    8 + i * 2,
                                    p.POSITION_CONTROL,
                                    targetPosition=1.362,
                                    force=self.finger1Force,
                                )
                        else:
                            for i in range(2):
                                p.setJointMotorControl2(
                                    self.jacoUid,
                                    8 + i * 2,
                                    p.POSITION_CONTROL,
                                    targetPosition=0.45,
                                    force=self.finger1Force,
                                )
                    else:
                        # 最后三位时电机角度值
                        for i in range(3):
                            p.setJointMotorControl2(
                                self.jacoUid,
                                8 + i * 2,
                                p.POSITION_CONTROL,
                                targetPosition=self.jacoFingerMotor[i],
                                force=self.finger1Force,
                            )

            elif self._jacoModel[:5] == "j2s7s":
                pass

    def setOriginalPosition(self):
        """
        @brief 配置机械臂到原始位置
        """
        # 是否使用确定位姿
        if self._useQuaternion:
            # 当位姿确定，机械臂夹爪始终向下
            jaco_original_position = list(
                self.accurateCalculateInverseKinematics(
                    self.jacoUid,
                    self.jacoEndEffectorIndex,
                    self._jacoEndEffectorPosition,
                    self.quaternion,
                )
            )
            # 末端夹爪的角度值只需要改变最后一个关节角度即可
            jaco_original_position[5] = 0
        else:
            # 当不使用确定位姿时，就直接进行逆运动学解算
            jaco_original_position = self.accurateCalculateInverseKinematics(
                self.jacoUid, self.jacoEndEffectorIndex, self._jacoEndEffectorPosition
            )
        if self._jacoModel[:5] == "j2n6s":
            # 机体
            for i in range(1, 7):
                p.setJointMotorControl2(
                    bodyIndex=self.jacoUid,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=jaco_original_position[i - 1],
                    force=self.maxForce,
                )
            # finger
            if self._jacoModel[5:8] == "200":
                for i in range(2):
                    p.setJointMotorControl2(
                        bodyIndex=self.jacoUid,
                        jointIndex=8 + i * 2,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=0.45,
                        force=self.finger1Force,
                    )
            elif self._jacoModel[5:8] == "300":
                for i in range(3):
                    p.setJointMotorControl2(
                        bodyIndex=self.jacoUid,
                        jointIndex=8 + i * 2,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=0.7,
                        force=self.finger1Force,
                    )
        elif self._jacoModel[:5] == "j2s7s":
            # TODO: 对于该类型机械臂控制代确定
            pass

    def accurateCalculateInverseKinematics(
        self,
        jacoUid,
        endEffectorIndex,
        targetPos,
        quaternion=None,
        threshold=0.0005,
        maxIter=50,
    ):
        """
        @brief 适用于jaco机械臂的精确的逆运动学解算
        @param jacoUid jaco机械臂id
        @param endEffectorIndex 末端关节id
        @param targetPos 目标位置
        @param quaternion 目标姿态
        @param threshold 定位精度，一般机器人的位置精度在1mm到5mm之间
        @param maxIter 最大迭代次数，为减少程序运行的时间，默认为50
        """
        closeEnough = False
        iter = 0
        dist2 = 1e8
        while not closeEnough and iter < maxIter:
            if quaternion is not None:
                jointPoses = p.calculateInverseKinematics(
                    jacoUid, endEffectorIndex, targetPos, quaternion
                )[:6]
            else:
                jointPoses = p.calculateInverseKinematics(
                    jacoUid, endEffectorIndex, targetPos
                )[:6]
            for i in range(1, 7):
                p.resetJointState(jacoUid, i, jointPoses[i - 1])
            ls = p.getLinkState(jacoUid, endEffectorIndex)
            newPos = ls[4]
            diff = [
                targetPos[0] - newPos[0],
                targetPos[1] - newPos[1],
                targetPos[2] - newPos[2],
            ]
            dist2 = np.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            closeEnough = dist2 < threshold
            iter = iter + 1
        # print("Num iter: " + str(iter) + "  threshold: " + str(dist2))
        return jointPoses
