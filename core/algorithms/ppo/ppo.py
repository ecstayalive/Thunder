import os
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from ...memory import RolloutBuffer
from .actor import Actor
from .critic import CriticV


class PPO:
    def __init__(
        self,
        env,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        action_scale: float = 1.0,
        lr: Tuple[float, float] = (5e-5, 5e-5),
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
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.action_scale = action_scale
        self.lr_actor = lr[0]
        self.lr_critic_v = lr[1]
        optimizer_dict = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD}
        self.optimizer = optimizer_dict[optimizer]
        # gradient clip
        self.max_grad_norm = 0.5
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
            print("Existing a ppo model, now, load it")
            self.load_model()
        else:
            self.critic_v = CriticV(self.obs_dim, device=self.training_device)
            self.actor = Actor(
                self.obs_dim,
                self.action_dim,
                action_scale=self.action_scale,
                device=self.training_device,
            )
            # training step, support checkpoint
            self.learning_step = 0
        # Initialize the optimizer
        self.actor_optimizer = self.optimizer(self.actor.parameters(), self.lr_actor)
        self.v_value_optimizer = self.optimizer(
            self.critic_v.parameters(), self.lr_critic_v
        )
        # Loss function
        self.loss_function = nn.MSELoss()

    def learn(
        self,
        max_learning_steps: int,
        rollout_steps: int = 256,
        learning_epochs: int = 10,
        mini_batch_size: int = 32,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
        evaluating_period: int = 1000,
        evaluating_times: int = 1,
    ) -> None:
        self.rollout_buffer = RolloutBuffer(rollout_steps)
        obs = self.env.reset()
        evaluation_mode_flag = False
        while self.learning_step < max_learning_steps:
            value = self.critic_v.calc_value(obs)
            action, action_log_prob = self.actor.act(obs)
            next_obs, reward, done, _ = self.env.step(action)
            next_value = self.critic_v.calc_value(next_obs)
            transition = {
                "observation": np.array(np.expand_dims(obs, 0), dtype=np.float32),
                "value": value,
                "action": np.array(np.expand_dims(action, 0), np.float32),
                "action_log_prob": np.array(
                    np.expand_dims(action_log_prob, 0), np.float32
                ),
                "reward": np.array(
                    [(reward + reward_bias) * reward_scale], dtype=np.float32
                ),
                "next_observation": np.array(
                    np.expand_dims(next_obs, 0), dtype=np.float32
                ),
                "next_value": next_value,
                "done": np.array([done], dtype=np.uint8),
            }
            self.rollout_buffer.store(transition)
            if done:
                obs = self.env.reset()
                if evaluation_mode_flag:
                    self.evaluate_model(evaluating_times)
                    evaluation_mode_flag = False
            else:
                obs = next_obs
            if self.rollout_buffer.length == rollout_steps:
                # compute the advantage
                self.rollout_buffer.compute_advantage(self.gamma, self.gae_lambda)
                # train the actor and v function model
                self.learn_model(learning_epochs, mini_batch_size)
                self.learning_step += 1
                # clear the rollout buffer
                self.rollout_buffer.clear()
                if self.learning_step % evaluating_period == 0:
                    evaluation_mode_flag = True

    def learn_model(self, epochs: int, batch_size: int) -> None:
        self.actor.train()
        self.critic_v.train()
        transitions_data = self.rollout_buffer.sample(batch_size)
        for key in transitions_data.keys():
            transitions_data[key] = torch.tensor(
                transitions_data[key], device=self.training_device
            )
        # calculate td target
        obs = transitions_data["observation"]
        value = transitions_data["value"]
        actions = transitions_data["action"]
        old_log_prob = transitions_data["action_log_prob"]
        advantage = transitions_data["advantage"]
        v_value_target = advantage + value

        # sample transitions
        for _ in range(epochs):
            # learning actor and critic
            new_log_prob = self.actor.calc_log_prob(obs, actions)
            ratio = torch.exp(new_log_prob - old_log_prob)
            surrogate_object1 = -ratio * advantage
            surrogate_object2 = (
                -torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                * advantage
            )
            actor_loss = torch.mean(torch.max(surrogate_object1, surrogate_object2))
            v_value_loss = torch.mean(
                self.loss_function(self.critic_v(obs), v_value_target)
            )
            # update actor parameter
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            self.v_value_optimizer.zero_grad()
            v_value_loss.backward()
            clip_grad_norm_(self.critic_v.parameters(), self.max_grad_norm)
            self.v_value_optimizer.step()
        # record the information of train
        self.writer.add_scalar("PPO/actor_loss", actor_loss.cpu(), self.learning_step)
        self.writer.add_scalar(
            "PPO/v_value_loss", v_value_loss.cpu(), self.learning_step
        )

    def evaluate_model(self, total_evaluating_steps: int = 100) -> None:
        self.env.is_test = True
        self.actor.eval()

        evaluating_step = 0
        sum_reward = 0

        obs = self.env.reset()
        while evaluating_step < total_evaluating_steps:
            action = self.actor.act(obs, evaluation_mode=True)
            next_obs, reward, done, _ = self.env.step(action)
            sum_reward += reward
            if done:
                obs = self.env.reset()
                evaluating_step += 1
            else:
                obs = next_obs

        self.writer.add_scalar(
            "PPO/mean_reward",
            np.array(sum_reward / total_evaluating_steps),
            self.learning_step,
        )
        self.env.is_test = False

    def calc_advantage(
        self, gamma: float, lam: float, td_delta: Tensor, done: int
    ) -> Tensor:
        """Calculate advantage of the episode"""
        advantage = td_delta.clone()
        episode_length = td_delta.shape[0]
        for idx in reversed(range(episode_length - 1)):
            advantage[idx] = (
                gamma * lam * advantage[idx + 1] * (1 - done[idx]) + td_delta[idx]
            )
        # advantage normalization
        advantage_mean = torch.mean(advantage)
        advantage_std = torch.std(advantage)
        advantage = (advantage - advantage_mean) / advantage_std

        return advantage

    def save_model(self):
        model = {
            "actor": self.actor,
            "critic_v": self.critic_v,
            "learning_step": self.learning_step,
        }

        torch.save(model, self.model_file_path)

    def load_model(self):
        model = torch.load(self.model_file_path, map_location=self.training_device)
        self.actor = model["actor"]
        self.critic_v = model["critic_v"]
        self.learning_step = model["learning_step"]
