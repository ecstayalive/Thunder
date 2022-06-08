"""
@author Bruce Hou
@brief 视觉伺服jaco机械臂抓取环境
@details 
@version 1.0
@date 2022-01-14
@email ecstayalive@163.com

@copyright Copyright (c) 2022
"""
import gym
from gym import spaces
from gym.utils import seeding
import pybullet as p
import pybullet_data as pd
import os, inspect
import numpy as np
import random
import glob
from Agent.Jaco2.jaco2 import Jaco2
from .setActionSpace import setActionSpace
from .getAction import getAction
import cv2
import warnings

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


class JacoVisionServoGraspEnv(gym.Env):
    """
    @brief Grasp environment: this would build a environment providing 
           all needs to train the robot grasping the unknown objects.
           It is a vision servo based grasp environment.
    """

    def __init__(
        self,
        jacoModel="j2n6s200",
        urdfRoot=pd.getDataPath(),
        actionRepeat=1,
        isEnableSelfCollision=True,
        render=True,
        isTest=False,
        blockRandom=0.1,
        useAngleSpace=False,
        numObjects=5,
        dv=0.15,
        maxStep=10,
        width=128,
        height=128,
        showImage=False,
        useDepthImage=False,
    ):
        self._jacoModel = jacoModel  # << jaco机械臂的型号
        self._urdfRoot = urdfRoot  # << pybullet自带的urdf文件路径
        self._timeStep = 1.0 / 240  # << 每一步的仿真时间
        self._isEnableSelfCollision = isEnableSelfCollision
        self._render = render  # << 是否启用渲染
        self._isTest = isTest
        self.maxForce = 200
        self.maxVelocity = 0.25
        self._blockRandom = blockRandom
        self._useAngleSpace = useAngleSpace
        self._actionRepeat = actionRepeat
        self._numObjects = numObjects
        self._dv = dv
        self._width = width
        self._height = height

        # 是否为视觉伺服
        self.vision_servo = True

        self._useDepthImage = useDepthImage

        self._maxSteps = maxStep

        self._showImage = showImage

        self._actionApplyTime = 300

        self.jacoEndEffectorIndex = 9

        self.useNullSpace = 21
        self.useOrientation = 1

        self._graspSuccess = 0  # << 抓取成功次数
        self._totalGraspTimes = 0  # << 尝试抓取次数
        # connect the physics engine
        if self._render:
            self.cid = p.connect(p.SHARED_MEMORY)
            if self.cid < 0:
                self.cid = p.connect(p.GUI)
            # set God view 上帝视角
            p.resetDebugVisualizerCamera(
                cameraDistance=0.85,
                cameraYaw=0,
                cameraPitch=-130,
                cameraTargetPosition=[0, 0, 0],
            )
        else:
            self.cid = p.connect(p.DIRECT)
        self.seed()

        self.reset()

    def reset(self):
        # set the environment of pybullet
        # 物理环境的设置
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)  # 求解迭代器的次数
        p.setTimeStep(self._timeStep)  # 时间步长
        # load object
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -0.65])
        p.loadURDF(
            os.path.join(self._urdfRoot, "table/table.urdf"), basePosition=[0, 0, -0.65]
        )
        p.loadURDF(
            "./Urdf/tray/tray.urdf", basePosition=[-0.1, 0, 0],
        )
        p.setGravity(0, 0, -9.81)

        # 设置机器人视角下的相机
        target_position = [0, 0, 0]
        # 使用球坐标
        camera_distance = 0.58
        camera_angle_gamma = 25 / 180 * np.pi
        camera_angle_theta = 160 / 180 * np.pi
        camera_position = [
            camera_distance * np.sin(camera_angle_gamma) * np.cos(camera_angle_theta),
            camera_distance * np.sin(camera_angle_gamma) * np.sin(camera_angle_theta),
            camera_distance * np.cos(camera_angle_gamma),
        ]
        up_vector = [
            np.sin(np.pi / 2 - camera_angle_gamma) * np.cos(camera_angle_theta + np.pi),
            np.sin(np.pi / 2 - camera_angle_gamma) * np.sin(camera_angle_theta + np.pi),
            np.cos(np.pi / 2 - camera_angle_gamma),
        ]
        self.viewMat = p.computeViewMatrix(
            cameraTargetPosition=target_position,
            cameraEyePosition=camera_position,
            cameraUpVector=up_vector,
        )
        # 视野大小
        fov = 80
        aspect = self._width / self._height
        near = 0.01
        far = 10
        self.projMat = p.computeProjectionMatrixFOV(fov, aspect, near, far)

        # 仿真步数
        self._env_step = 0
        self._termination = False

        # set the robotic arm
        self.jaco2 = Jaco2(
            jacoModel=self._jacoModel,
            basePosition=[0.5, 0, 0],
            jacoEndEffectorPosition=[0, 0, 0.3],
            useAngleSpace=self._useAngleSpace,  # << 默认不使用空间角度
            useQuaternion=True,  # << 默认使机械臂夹爪竖直向下
        )

        # 观测空间的设置
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(1, self._height, self._width), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(5,)
        )  # dx, dy, dz, d_anger, d_gripper

        # Choose the objects in the bin.
        self._numObjects = np.random.randint(1, 6)
        urdfList = self.getRandomObjects(self._numObjects, self._isTest)
        self._objectUids = self.randomlyPlaceObjects(urdfList)
        self._observation = self.getObservation()
        return self._observation

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        dv = self._dv  # velocity per physics step
        if self._useAngleSpace:
            pass
        else:
            dx = action[0] * dv
            dy = action[1] * dv
            dz = action[2] * dv
            dtheta = action[3] * 0.25
            gripper_state = np.heaviside(action[4], 0)
            action = [dx, dy, dz, dtheta, gripper_state]

        return self.stepContinuous(action)

    def stepContinuous(self, action):
        """
        @brief 
        """
        # Perform comanded action
        self._env_step += 1

        # Apply action
        self.jaco2.applyAction(action)

        observation = self.getObservation()
        done = self.terminate()
        reward = self.reward()

        # 在抓取成功后，确定是否需要进行下一轮的抓取，进行下一轮的抓取
        if reward == 1.0:
            done = True

        debug = {"grasp_success": self._graspSuccess}
        if done:
            self._totalGraspTimes += 1
            if self._totalGraspTimes == 0:
                print(f"\nreward:{reward}, done:{done}, info:{debug}")
            else:
                print(
                    f"\nreward:{reward}, done:{done}, info:{debug}, total grasp times:{self._totalGraspTimes}"
                )

        return observation, reward, done, debug

    def reward(self):
        """
        @brief 奖励函数
        @details 通过改变奖励函数改变机器人表现
                 目前在规定步数之内抓取成功, 则奖励为1, 否则为0
                 若直到最后一次都没有抓取成功，则奖励为-0.01
        """
        reward = 0

        # TODO: 如何使用帧差分法来判断抓取成功，并给出奖励
        # TODO: 如果想要支持多个物品一个一个的抓取出来，那么就不需要break，最后的奖励应该是抓取成功的次数
        for uid in self._objectUids:
            pos, _ = p.getBasePositionAndOrientation(uid)
            # If any block is above height, provide reward.
            if pos[2] > 0.2:
                self._graspSuccess += 1
                reward = 1
                break

        return reward

    def randomlyPlaceObjects(self, urdfList):
        """
        @brief 随机放置物体
        """
        objectUids = []
        for urdf_name in urdfList:
            xpos = 0 + self._blockRandom * np.random.uniform(-1, 1)
            ypos = 0 + 2 * self._blockRandom * np.random.uniform(-1, 1)
            angle = np.pi / 2 + self._blockRandom * np.pi * random.random()
            orn = p.getQuaternionFromEuler([0, 0, angle])
            urdf_path = os.path.join(self._urdfRoot, urdf_name)
            # print(urdf_name)
            uid = p.loadURDF(
                urdf_path, [xpos, ypos, 0.2], [orn[0], orn[1], orn[2], orn[3]]
            )
            objectUids.append(uid)
            # 进行仿真
            for _ in range(self._actionApplyTime):
                p.stepSimulation()
        return objectUids

    def getObservation(self):
        """
        @brief 获取当前步的相机图像
        """
        # View state
        (_, _, px, dx, _) = p.getCameraImage(
            width=self._width,
            height=self._height,
            viewMatrix=self.viewMat,
            projectionMatrix=self.projMat,
            # renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        self.rgbImage = np.array(px, dtype=np.uint8)[:, :, :3][:, :, ::-1]
        self.rgbImage = cv2.cvtColor(self.rgbImage, cv2.COLOR_BGR2GRAY)
        if self._showImage:
            img = self.rgbImage.copy()
            img = cv2.resize(img, (240, 240), cv2.INTER_AREA)
            cv2.imshow("observation", img)
            cv2.waitKey(1)
        self.rbgImage = np.array(self.rgbImage, dtype=np.float32)
        obs = np.array(np.expand_dims(self.rgbImage, axis=0) / 255.0, dtype=np.float32)

        return obs

    def getRandomObjects(self, num_objects, test):
        """
        @brief 获取被放在目标区域的随机物品.
        """
        if test:
            urdf_pattern = os.path.join(self._urdfRoot, "random_urdfs/*0/*.urdf")
        else:
            urdf_pattern = os.path.join(self._urdfRoot, "random_urdfs/*[1-9]/*.urdf")
        found_object_directories = glob.glob(urdf_pattern)
        total_num_objects = len(found_object_directories)
        selected_objects = np.random.choice(np.arange(total_num_objects), num_objects)
        selected_objects_filenames = []
        for object_index in selected_objects:
            selected_objects_filenames += [found_object_directories[object_index]]
        return selected_objects_filenames

    def terminate(self):
        """
        @brief 终止函数
        @details 终止函数，用于终止程序
        """
        if self._env_step == self._maxSteps:
            self._termination = True
            return True
        else:
            return False

    def close(self):
        """
        @brief 关闭仿真环境
        """
        p.disconnect()
