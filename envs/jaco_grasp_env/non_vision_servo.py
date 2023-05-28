"""Grasp environment
This would build a environment providing all needs to
train the robot grasping the unkonwn objects. Through
it is a non-vision servo based environment, it still
is based on vision.

"""

import pybullet as p
from gymnasium import spaces

from .framework import JacoGraspEnvFramework


class JacoNonVisionServoGraspEnv(JacoGraspEnvFramework):
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
        super(JacoNonVisionServoGraspEnv, self).__init__(
            jaco_model,
            render,
            is_test,
            block_random,
            dv,
            max_step,
            width,
            height,
            show_image,
            use_depth_image,
        )
        self.env_name = "JacoNonVisionServoGrasp"
        self.vision_servo = False
        ########################################################################
        # action spaces
        ########################################################################
        self.action_space = spaces.Box(low=-10, high=10, shape=(3,))  # dx, dy, dangle
        self.seed()

    def reset(self):
        self.attempted_grasp = False
        return self.env_reset()

    def step(self, action):
        # velocity per physics step
        dv = self.dv
        dx = dv * action[0]
        dy = dv * action[1]
        dz = -dv
        da = action[2]
        action = [dx, dy, dz, da, 1]
        # Perform comanded action
        self.env_step += 1
        # Apply action
        self.jaco2.apply_action(action)
        for _ in range(500):
            p.stepSimulation()
        # 如果机械臂靠近物体，就抓取物体
        state = self.jaco2.get_end_effector_state()
        end_effector_pos = state[0]
        # 满足一些条件尝试抓取
        if end_effector_pos[2] <= 0.07:
            self.apply_grasp_action()
            self.attempted_grasp = True
        observation = self.get_observation()
        # 获取奖励
        reward = self.reward()
        # 判断是否终止程序
        done = self.terminate()
        info = {"successful_times": self.successful_times}
        if done:
            self.total_grasp_times += 1
            if self.total_grasp_times == 0:
                print(f"\nreward:{reward}, done:{done}, info:{self.successful_times}")
            else:
                print(
                    f"\nreward:{reward}, done:{done}, successful_grasp_times:{self.successful_times}, total grasp times:{self.total_grasp_times}"
                )
        return observation, reward, done, info

    def reward(self):
        """Reward function
        通过改变奖励函数改变机器人表现
        目前是抓取成功奖励为1,其余为0

        Returns:
            reward

        """
        reward = 0
        if self.attempted_grasp:
            for uid in self.object_uids:
                pos, _ = p.getBasePositionAndOrientation(uid)
                # If any block is above height, provide reward.
                if pos[2] > 0.2:
                    self.successful_times += 1
                    reward = 100.0
                    break

        return reward

    def terminate(self):
        """Terminating function
        终止函数，用于终止程序

        """
        return bool(self.attempted_grasp or self.env_step == self.max_step)

    def apply_grasp_action(self):
        finger_angle = 1
        for _ in range(300):
            grasp_action = [0, 0, 0, 0, finger_angle]
            self.jaco2.apply_action(grasp_action)
            p.stepSimulation()
            finger_angle -= 1 / 100.0
            finger_angle = max(finger_angle, 0)
        for _ in range(1000):
            grasp_action = [0, 0, 0.002, 0, finger_angle]
            self.jaco2.apply_action(grasp_action)
            p.stepSimulation()
            finger_angle -= 1 / 100.0
            finger_angle = max(finger_angle, 0)
