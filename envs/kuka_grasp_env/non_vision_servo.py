import pybullet as p
from gymnasium import spaces

from .framework import KukaGraspEnvFramework


class KukaNonVisionServoGraspEnv(KukaGraspEnvFramework):
    """Class for Kuka environment with diverse objects.
    In each episode some objects are chosen from a set of 1000 diverse objects.
    These 1000 objects are split 90/10 into a train and test set.
    """

    def __init__(
        self,
        render=True,
        is_test=False,
        block_random=0.3,
        dv=0.06,
        max_step=10,
        camera_random=0,
        width=128,
        height=128,
        show_image=True,
        use_depth_image=False,
    ):
        super(KukaNonVisionServoGraspEnv, self).__init__(
            render,
            is_test,
            block_random,
            dv,
            max_step,
            camera_random,
            width,
            height,
            show_image,
            use_depth_image,
        )
        self.env_name = "KukaNonVisionServoGrasp"
        # 是否为视觉伺服
        self.vision_servo = False
        ########################################################################
        # action spaces
        ########################################################################
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,))  # dx, dy, dangle

        self.total_grasp_times = 0
        self.successful_grasp_times = 0

    def reset(self):
        """Environment reset called at the beginning of an episode."""
        self.attempted_grasp = False
        return self.env_reset()

    def step(self, action):
        """Environment step.
        Args:
            action: 5-vector parameterizing XYZ offset, vertical angle offset
            (radians), and grasp angle (radians).
        Returns:
            observation: Next observation.
            reward: Float of the per-step reward as a result of taking the action.
            done: Bool of whether or not the episode has ended.
            debug: Dictionary of extra information provided by environment.
        """
        dv = self.dv  # velocity per physics step.
        dx = dv * action[0]
        dy = dv * action[1]
        dz = -dv
        da = 0.25 * action[2]
        action = [dx, dy, dz, da, 1]

        # Perform commanded action.
        self.env_step += 1
        self.kuka.apply_action(action)
        for _ in range(300):
            p.stepSimulation()

        # If we are close to the bin, attempt grasp.
        state = self.kuka.get_end_effector_state()
        end_effector_pos = state[0]
        if end_effector_pos[2] <= 0.1:
            self.apply_grasp_action()
            self.attempted_grasp = True
        observation = self.get_observation()
        done = self.terminate()
        reward = self.reward()
        if reward > 0.8:
            self.successful_grasp_times += 1

        info = {"successful_grasp_times": self.successful_grasp_times}
        if done:
            self.total_grasp_times += 1
            print(
                f"done: {done}, \
                reward: {reward}, \
                successful_grasp_times: {self.successful_grasp_times}, \
                total_grasp_times: {self.total_grasp_times}"
            )
        return observation, reward, done, info

    def reward(self):
        """Calculates the reward for the episode.

        The reward is 1 if one of the objects is above height .2 at the end of the
        episode.
        """
        reward = 0
        for uid in self.object_uids:
            pos, _ = p.getBasePositionAndOrientation(uid)
            # If any block is above height, provide reward.
            if pos[2] > 0.15:
                reward = 100
                break
        return reward

    def terminate(self):
        """
        Terminates the episode if we have tried to grasp or if we are above max steps.
        """
        return self.attempted_grasp or self.env_step >= self.max_step

    def apply_grasp_action(self):
        finger_angle = 1
        for _ in range(500):
            grasp_action = [0, 0, 0, 0, finger_angle]
            self.kuka.apply_action(grasp_action)
            p.stepSimulation()
            finger_angle -= 1 / 100.0
            finger_angle = max(finger_angle, 0)
        for _ in range(500):
            grasp_action = [0, 0, 0.001, 0, finger_angle]
            self.kuka.apply_action(grasp_action)
            p.stepSimulation()
            finger_angle -= 1 / 100.0
            finger_angle = max(finger_angle, 0)
