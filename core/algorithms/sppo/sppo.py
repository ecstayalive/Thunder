import os
from typing import Callable, Tuple

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

from ...memory import RolloutBuffer
from .actor import Actor
from .critic import CriticV


class SPPO:
    def __init__(
        self,
        env,
        gamma: float = 0.99,
        lam: float = 0.95,
        epsilon: float = 0.1,
        tau: float = 5e-3,
        action_scale: float = 1.0,
        lr: float = 3e-4,
        optimizer="Adam",
        logfile_dir: str = "log/",
    ) -> None:
        # get some parameters of env
        self.env = env
        self.action_dim = env.action_space.shape[0]
        if len(env.observation_space.shape) == 1:
            self.obs_dim = env.observation_space.shape[0]
        else:
            self.obs_dim = env.observation_space.shape
        # the target entropy
        self.target_entropy = -self.action_dim
        # configure files path
        self.configure_files_path(env, logfile_dir)

        # some parameters about training
        self.training_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.tau = tau
        self.action_scale = action_scale
        self.lr_actor = lr[0]
        self.lr_critic_q = lr[1]
        self.lr_log_alpha = lr[2]
        optimizer_dict = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD}
        self.optimizer = optimizer_dict[optimizer]
        # gradient clip
        self.grad_max_norm = 1
        self.minimal_buffer_length = 1000
        # rollout buffer
        self.rollout_buffer = RolloutBuffer(env.max_episode_steps)
        # record training information
        self.writer = SummaryWriter(log_dir=self.train_logfile_dir)
        # create models including policy, value function and optimizer
        self.create_model()

    def configure_files_path(self, env: Callable, logfile_dir: str) -> None:
        """Set all log files path.
        Firstly, set all log files path and names. Secondly,
        Check whether the model save path exists. If not,
        create the path.

        Args:
            env: the simulation environment
            logfile_dir:
        """
        self.model_save_dir = logfile_dir + env.env_name + "/ppo/model/"
        self.model_file_path = logfile_dir + env.env_name + "/ppo/model/ppo_model.pkl"
        self.train_logfile_dir = logfile_dir + env.env_name + "/ppo/train_log/"
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

    def create_model(self) -> None:
        """Load or create model
        If there is a model, then load it, if not, create a new one.

        """
        if os.path.exists(self.model_file_path):
            print("Existing a sac model, now, load it")
            self.load_model()
        else:
            self.critic_v = CriticV(
                self.obs_dim, self.action_dim, device=self.training_device
            )
            self.target_critic_v = CriticV(
                self.obs_dim, self.action_dim, device=self.training_device
            )
            self.actor = Actor(
                self.obs_dim,
                self.action_dim,
                action_scale=self.action_scale,
                device=self.training_device,
            )
            # Initialize the target v net
            self.target_critic_v.load_state_dict(self.critic_v.state_dict())
            # training step, support checkpoint
            self.learning_step = 0
        # Initialize the optimizer
        self.actor_optimizer = self.optimizer(self.actor.parameters(), self.lr_actor)
        self.v_value_optimizer = self.optimizer(
            self.critic_v.parameters(), self.lr_critic_q
        )
        # Loss function
        self.loss_function = nn.MSELoss()

    def learn(
        self,
        max_learning_steps: int,
        learning_epochs: int = 10,
        sample_batch_size: int = 64,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
        evaluating_period: int = 1000,
        evaluating_times: int = 1,
    ) -> None:
        while self.learning_step < max_learning_steps:
            obs = self.env.reset()
            done = False
            # clear the rollout buffer
            self.rollout_buffer.clear()
            while not done:
                action = self.actor.act(obs)
                next_obs, reward, done, _ = self.env.step(action)
                transition = {
                    "observation": np.array(np.expand_dims(obs, 0), dtype=np.float32),
                    "action": np.array(np.expand_dims(action, 0), np.float32),
                    "reward": np.array(
                        [(reward + reward_bias) * reward_scale], dtype=np.float32
                    ),
                    "next_observation": np.array(
                        np.expand_dims(next_obs, 0), dtype=np.float32
                    ),
                    "done": np.array([done], dtype=np.uint8),
                }
                self.rollout_buffer.store(transition)
            for _ in learning_epochs:
                self.learn_model(sample_batch_size)
                self.learning_step += 1
                if self.learning_step % evaluating_period:
                    self.evaluate_model(evaluating_times)

    def learn_model(self, epochs: int, batch_size: int) -> None:
        self.actor.train()
        self.critic_v.train()

        episode = self.rollout_buffer.get_data()

    def calculate_td_error(self):
        ...

    def calculate_advantage(self, gamma, lam, td_error):
        ...
