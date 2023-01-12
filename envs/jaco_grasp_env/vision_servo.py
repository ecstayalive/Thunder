"""Grasp environment
This would build a environment providing all needs to 
train the robot grasping the unknown objects.It is a 
vision servo based grasp environment.

"""

import pybullet as p
from gym import spaces
from .framework import JacoGraspEnvFramework


class JacoVisionServoGraspEnv(JacoGraspEnvFramework):
    def __init__(
        self,
        jaco_model="j2n6s200",
        render=True,
        is_test=False,
        block_random=0.1,
        num_objects=5,
        dv=0.15,
        max_step=10,
        width=128,
        height=128,
        show_image=False,
        use_depth_image=False,
    ):
        super(JacoVisionServoGraspEnv, self).__init__(
            jaco_model,
            render,
            is_test,
            block_random,
            num_objects,
            dv,
            max_step,
            width,
            height,
            show_image,
            use_depth_image,
        )
        # 是否为视觉伺服
        self.vision_servo = True
        self.seed()
        self.reset()

    def reset(self):
        ########################################################################
        # action spaces
        ########################################################################
        self.action_space = spaces.Box(
            low=-10, high=10, shape=(5,)
        )  # dx, dy, dz, dangle, finger
        return self.env_reset()

    def step(self, action):
        dv = self.dv  # velocity per physics step
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        dtheta = action[3]
        finger_angle = action[4] + 0.5
        action = [dx, dy, dz, dtheta, finger_angle]
        # Perform comanded action
        self.env_step += 1
        # Apply action
        self.jaco2.apply_action(action)
        for _ in range(800):
            p.stepSimulation()
        observation = self.get_observation()
        # 获取奖励
        reward = self.reward()
        # 判断是否终止程序
        done = self.terminate()

        # 在抓取成功后，确定是否需要进行下一轮的抓取，进行下一轮的抓取
        if reward == 1.0:
            done = True
            self.successful_grasp_times += 1

        info = {"grasp_success": self.successful_grasp_times}
        if done:
            self.total_grasp_times += 1
            if self.total_grasp_times == 0:
                print(f"\nreward:{reward}, done:{done}, info:{info}")
            else:
                print(
                    f"\nreward:{reward}, done:{done}, successful_grasp_times:{self.successful_grasp_times}, total grasp times:{self.total_grasp_times}"
                )

        return observation, reward, done, info

    def reward(self):
        reward = 0
        # TODO(ecstayalive@163.com): 使用帧差分法来判断抓取成功，并给出奖励
        # NOTE(ecstayalive@163.com): 如果想要支持多个物品一个一个的抓取出来，那么就不需要break，最后的奖励应该是抓取成功的次数
        for uid in self.object_uids:
            pos, _ = p.getBasePositionAndOrientation(uid)
            # If any block is above height, provide reward.
            if pos[2] > 0.2:
                self.successful_grasp_times += 1
                reward = 1
                break

        return reward

    def terminate(self):
        if self.env_step == self.max_step:
            return True
        else:
            return False
