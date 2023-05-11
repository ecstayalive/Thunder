import glob
import os
import random
from abc import ABC, abstractmethod

import cv2
import gym
import numpy as np
import pybullet as p
import pybullet_data as pd
from agent import Jaco2
from gym import spaces
from gym.utils import seeding


class JacoGraspEnvFramework(
    gym.Env,
    ABC,
):
    """Jaco robotic arm grasp envs' framework."""

    def __init__(
        self,
        jaco_model="j2n6s200",
        render=True,
        is_test=False,
        block_random=0.2,
        dv=0.06,
        max_step=10,
        width=256,
        height=256,
        show_image=False,
        use_depth_image=False,
    ):
        """Initialization

        Args:
            jaco_model:
            render:
            is_test:
            block_random:
            dv:
            max_step:
            width:
            height:
            show_image:
            use_depth_image:

        """
        super(JacoGraspEnvFramework, self).__init__()
        self.jaco_model = jaco_model  # << jaco机械臂的型号
        self.urdf_root = pd.getDataPath()  # << pybullet自带的urdf文件路径
        self.time_step = 1.0 / 240  # << 每一步的仿真时间
        self.is_test = is_test
        self.max_force = 500
        self.max_velocity = 0.25
        self.block_random = block_random
        self.dv = dv
        self.width = width
        self.height = height
        self.vision_servo = False
        self.use_depth_image = use_depth_image
        self.max_step = max_step
        self.show_image = show_image
        # several parameters
        self.action_apply_time = 500
        self.successful_times = 0  # << 抓取成功次数
        self.total_grasp_times = 0  # << 尝试抓取次数

        # connect the physics engine
        if render:
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
        ########################################################################
        # observation spaces
        ########################################################################
        if self.use_depth_image:
            pass
        else:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(1, self.height, self.width), dtype=np.float32
            )

    @abstractmethod
    def reset(self):
        """please implement in subclass"""

    def env_reset(self):
        ########################################################################
        # set the environment of pybullet
        ########################################################################
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)  # 求解迭代器的次数
        p.setTimeStep(self.time_step)  # 时间步长
        # load objects
        p.loadURDF(os.path.join(self.urdf_root, "plane.urdf"), [0, 0, -0.65])
        p.loadURDF(
            os.path.join(self.urdf_root, "table/table.urdf"), basePosition=[0, 0, -0.65]
        )
        p.loadURDF(
            "./urdf/tray/tray.urdf",
            basePosition=[-0.1, 0, 0],
        )
        p.setGravity(0, 0, -9.81)
        ########################################################################
        # set camera
        ########################################################################
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
        self.view_mat = p.computeViewMatrix(
            cameraTargetPosition=target_position,
            cameraEyePosition=camera_position,
            cameraUpVector=up_vector,
        )
        ########################################################################
        # set field of view size
        ########################################################################
        fov = 80
        aspect = self.width / self.height
        near = 0.01
        far = 10
        self.proj_mat = p.computeProjectionMatrixFOV(fov, aspect, near, far)
        ########################################################################
        # load the robotic arm
        ########################################################################
        self.jaco2 = Jaco2(
            jaco_model=self.jaco_model,
            base_position=[0.5, 0, 0],
            jaco_end_effector_position=[0, 0, 0.3],
        )
        ########################################################################
        # set parameters
        ########################################################################
        # 仿真步数
        self.env_step = 0
        ########################################################################
        # Choose the objects in the bin
        ########################################################################
        self.num_objects = np.random.randint(1, 6)
        urdf_list = self.get_random_objects(self.num_objects, self.is_test)
        self.object_uids = self.place_objects_randomly(urdf_list)
        self.observation = self.get_observation()
        return self.observation

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @abstractmethod
    def step(self, action):
        """please implement in subclass"""

    @abstractmethod
    def reward(self):
        """Reward function
        通过改变奖励函数改变机器人表现
        目前是抓取成功奖励为1,其余为0

        Returns:
            reward

        """

        """please implement in subclass"""

    def place_objects_randomly(self, urdf_list):
        """Place objects randomly"""
        # Randdomize positions of each object urdf
        object_uids = []
        for urdf_name in urdf_list:
            xpos = 0 + 2 * self.block_random * random.random() - self.block_random
            ypos = 0 + 2 * self.block_random * random.random() - self.block_random
            angle = np.pi / 2 + self.block_random * np.pi * random.random()
            orn = p.getQuaternionFromEuler([0, 0, angle])
            urdf_path = os.path.join(self.urdf_root, urdf_name)
            # print(urdf_name)
            uid = p.loadURDF(
                urdf_path, [xpos, ypos, 0.2], [orn[0], orn[1], orn[2], orn[3]]
            )
            object_uids.append(uid)
            # Let each object fall to the tray individual, to prevent object
            # intersection.
            for _ in range(self.action_apply_time):
                p.stepSimulation()
        return object_uids

    def get_observation(self):
        """获取当前步的相机图像"""
        # View state
        (_, _, px, dx, _) = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_mat,
            projectionMatrix=self.proj_mat,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        self.rgb_image = np.array(px, dtype=np.uint8)[:, :, :3][:, :, ::-1]
        self.gray_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)
        if self.show_image:
            img = self.rgb_image.copy()
            img = cv2.resize(img, (256, 256), cv2.INTER_AREA)
            cv2.imshow("observation", img)
            cv2.waitKey(1)
        self.gray_image = np.array(self.gray_image, dtype=np.float32)
        return np.array(
            np.expand_dims(self.gray_image, axis=0) / 255.0, dtype=np.float32
        )

    def get_random_objects(self, num_objects, test):
        """Randomly choose an object urdf from the random_urdfs directory."""
        if test:
            urdf_pattern = os.path.join(self.urdf_root, "random_urdfs/*0/*.urdf")
        else:
            urdf_pattern = os.path.join(self.urdf_root, "random_urdfs/*[1-9]/*.urdf")
        found_object_directories = glob.glob(urdf_pattern)
        total_num_objects = len(found_object_directories)
        selected_objects = np.random.choice(np.arange(total_num_objects), num_objects)
        return [
            found_object_directories[object_index] for object_index in selected_objects
        ]

    @abstractmethod
    def terminate(self):
        """Terminating function
        终止函数，用于终止程序

        """

        """please implement in subclass"""

    def close(self):
        """Close simulation environment"""
        p.disconnect()
