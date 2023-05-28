from typing import Optional

import numpy as np
from gymnasium.envs.classic_control import PendulumEnv as _PendulumEnv


class PendulumEnv(_PendulumEnv):
    def __init__(
        self,
        render_mode: Optional[str] = None,
        g: float = 9.81,
        max_episode_steps: int = 200,
    ):
        super().__init__(render_mode, g)
        self.env_name = "Pendulum"
        self.is_test = False
        self.max_episode_steps = max_episode_steps

    def step(self, action: float):
        action = np.array(action, np.float32)
        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated | truncated
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
