import glob
import os
import random
from abc import ABC, abstractmethod

import cv2
import gym
import numpy as np
import pybullet as p
import pybullet_data as pd
from agent import Kuka
from gym import spaces
from gym.utils import seeding


class KukaGraspEnvFramework(
    gym.Env,
    ABC,
):
    """Kuka robotic arm grasp envs' framework."""

    def __init__(
        self,
        render=True,
        is_test=False,
        block_random=0.2,
        dv=0.1,
        max_step=10,
        camera_random=0,
        width=128,
        height=128,
        show_image=False,
        use_depth_image=False,
    ):
        """Initializes the KukaDiverseObjectEnv.

        Args:
            renders: If true, render the bullet GUI.
            is_test: If true, use the test set of objects. If false, use the train
                set of objects.
            block_random: A float between 0 and 1 indicated block randomness. 0 is
                deterministic.
            dv: The velocity along each dimension for each action.
            max_step: The maximum number of actions per episode.
            camera_random: A float between 0 and 1 indicating camera placement
                randomness. 0 is deterministic.
            width: The image width.
            height: The observation image height.
            num_objects: The number of objects in the bin.
            show_image:
            use_depth_image:

        """
        super(KukaGraspEnvFramework, self).__init__()
        self.urdf_root = pd.getDataPath()  # << pybullet自带的urdf文件路径
        self.time_step = 1.0 / 240  # << 每一步的仿真时间
        self.env_step = 0
        self.is_test = is_test
        self.max_force = 500
        self.max_velocity = 0.25
        self.block_random = block_random
        self.dv = dv
        self.camera_random = camera_random
        self.width = width
        self.height = height
        self.vision_servo = False
        self.use_depth_image = use_depth_image
        self.max_step = max_step
        self.show_image = show_image
        # several parameters
        self.action_apply_time = 500
        self.successful_grasp_times = 0  # << 抓取成功次数
        self.total_grasp_times = 0  # << 尝试抓取次数

        # connect the physics engine
        if render:
            self.cid = p.connect(p.SHARED_MEMORY)
            if self.cid < 0:
                self.cid = p.connect(p.GUI)
            # set God view 上帝视角
            p.resetDebugVisualizerCamera(
                cameraDistance=1.3,
                cameraYaw=180,
                cameraPitch=-41,
                cameraTargetPosition=[0.52, -0.2, -0.33],
            )
        else:
            self.cid = p.connect(p.DIRECT)
        self.seed()
        ########################################################################
        # observation spaces
        ########################################################################
        if self.use_depth_image:
            pass
        else:
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(3, self.height, self.width), dtype=np.float32
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
        p.loadURDF(os.path.join(self.urdf_root, "plane.urdf"), [0, 0, -1])
        p.loadURDF(
            os.path.join(self.urdf_root, "table/table.urdf"),
            0.5000000,
            0.00000,
            -0.820000,
            0.000000,
            0.000000,
            0.0,
            1.0,
        )
        p.setGravity(0, 0, -9.81)
        ########################################################################
        # load block
        ########################################################################
        self.tray_uid = p.loadURDF(
            os.path.join(self.urdf_root, "tray/tray.urdf"),
            0.640000,
            0.075000,
            -0.190000,
            0.000000,
            0.000000,
            1.000000,
            0.000000,
        )
        ########################################################################
        # load kuka
        ########################################################################
        self.kuka = Kuka(time_step=self.time_step)
        ########################################################################
        # set camera
        ########################################################################
        # TODO(ecstayalive@163.com): optimize the camera locate position
        target_position = [0.23, 0.2, 0.54]
        distance = 0.5
        pitch = -56 + self.camera_random * np.random.uniform(-3, 3)
        yaw = 245 + self.camera_random * np.random.uniform(-3, 3)
        roll = 0
        self.view_mat = p.computeViewMatrixFromYawPitchRoll(
            target_position, distance, yaw, pitch, roll, 2
        )
        fov = 20.0 + self.camera_random * np.random.uniform(-2, 2)
        aspect = self.width / self.height
        near = 0.01
        far = 10
        self.proj_mat = p.computeProjectionMatrixFOV(fov, aspect, near, far)

        ########################################################################
        # set field of view size
        ########################################################################
        fov = 60
        aspect = self.width / self.height
        near = 0.1
        far = 100
        self.proj_mat = p.computeProjectionMatrixFOV(fov, aspect, near, far)
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
        # Randomize positions of each object urdf.
        object_uids = []
        for urdf_name in urdf_list:
            xpos = 0.4 + self.block_random * random.random()
            ypos = self.block_random * (random.random() - 0.5)
            angle = np.pi / 2 + self.block_random * np.pi * random.random()
            orn = p.getQuaternionFromEuler([0, 0, angle])
            urdf_path = os.path.join(self.urdf_root, urdf_name)
            uid = p.loadURDF(
                urdf_path, [xpos, ypos, 0.15], [orn[0], orn[1], orn[2], orn[3]]
            )
            object_uids.append(uid)
            # Let each object fall to the tray individual, to prevent object
            # intersection.
            for _ in range(500):
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
        )
        self.rgb_image = np.array(px, dtype=np.uint8)[:, :, :3][:, :, ::-1]
        # self.rgb_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)
        if self.show_image:
            img = self.rgb_image.copy()
            cv2.imshow("observation", img)
            cv2.waitKey(1)
        self.rbg_image = np.array(self.rgb_image / 255.0, dtype=np.float32)
        return self.rbg_image.transpose(2, 0, 1)

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
