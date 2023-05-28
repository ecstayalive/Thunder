from typing import Optional

import numpy as np
from gymnasium.envs.classic_control import Continuous_MountainCarEnv


class ContinuousMountainCarEnv(Continuous_MountainCarEnv):
    def __init__(
        self,
        render_mode: Optional[str] = None,
        goal_velocity: float = 0.0,
        max_episode_steps: int = 999,
        reward_scale: float = 1.0,
    ):
        super().__init__(render_mode, goal_velocity)
        self.env_name = "ContinuousMountainCar"
        self.is_test = False
        self.max_episode_steps = max_episode_steps
        self.reward_scale = reward_scale

    def step(self, action: float):
        action = np.array(action, np.float32)
        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated | truncated
        if not self.is_test:
            reward *= self.reward_scale
        self.episodes_length += 1
        if self.episodes_length >= self.max_episode_steps:
            done = True
        return obs, reward, done, info

    def reset(self):
        initialized_obs, _ = super().reset()
        self.episodes_length = 0
        return initialized_obs

    def close(self):
        super().close()
