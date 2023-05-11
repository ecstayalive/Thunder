import os
from typing import Callable, Tuple

import numpy as np

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from ...memory import ReplyBuffer
from .actor import Actor
from .critic import CriticQ, CriticV


class SAC:
    def __init__(
        self,
        env: Callable,
        buffer_size: int = 10000,
        gamma: float = 0.98,
        temperature_factor: float = 0.8,
        lr_actor: float = 1e-5,
        lr_critic: Tuple[float, float] = [1e-5, 1e-5],
        sample_batch_size: int = 16,
        model_file_path: str = "model/sac_model.pkl",
        logfile_dir: str = "log/sac/",
    ) -> None:
        # initialization
        self.model_file_path = model_file_path
        self.logfile_dir = logfile_dir

        # get some parameters of env
        self.env = env
        self.action_dim = env.action_space.shape[0]
        self.obs_dim = env.observation_space.shape

        # some parameters about training
        self.training_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.gamma = gamma
        self.temperature_factor = temperature_factor
        self.lr_actor = lr_actor
        self.lr_critic_q = lr_critic[0]
        self.lr_critic_v = lr_critic[1]
        self.tau = 1e-3
        # self.clip_grad = 1

        # some frequency
        self.net_analysis_iter = 1000
        self.learning_iter = 20
        self.evaluate_iter = 1000
        self.repeat_learning_times = 40

        # buffer setting
        self.reply_buffer = ReplyBuffer()
        self.buffer_capacity = buffer_size
        self.sample_batch_size = sample_batch_size

        # record training information
        self.writer = SummaryWriter(log_dir=logfile_dir)
        self.create_model()

    def create_model(self) -> None:
        """Load or create model
        If there is a model, then load it, if not, create a new one.

        """
        if os.path.exists(self.model_file_path):
            print("Existing a sac model, now, load it")
            self.load_model()
        else:
            self.critic_q = CriticQ(
                self.obs_dim, self.action_dim, device=self.training_device
            )
            self.critic_v = CriticV(self.obs_dim, device=self.training_device)
            self.target_critic_v = CriticV(self.obs_dim, device=self.training_device)
            self.actor = Actor(
                self.obs_dim, self.action_dim, device=self.training_device
            )
            # Initialize the target net
            for target_critic_v_param, critic_v_param in zip(
                self.target_critic_v.parameters(), self.critic_v.parameters()
            ):
                target_critic_v_param.data.copy_(critic_v_param.data)
            # training step, support checkpoint
            self.learning_step = 0
        # Initialize the optimizer
        # Use SGD
        self.actor_optimizer = torch.optim.SGD(self.actor.parameters(), self.lr_actor)
        self.q_value_optimizer = torch.optim.SGD(
            self.critic_q.parameters(), self.lr_critic_q
        )
        self.v_value_optimizer = torch.optim.SGD(
            self.critic_v.parameters(), self.lr_critic_v
        )
        # # Use Adam
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.lr_actor)
        # self.q_value_optimizer = torch.optim.Adam(
        #     self.critic_q.parameters(), self.lr_critic_q
        # )
        # self.v_value_optimizer = torch.optim.Adam(
        #     self.critic_v.parameters(), self.lr_critic_v
        # )

        # Loss function
        self.loss_function = nn.MSELoss()

    def learn(self, total_learning_steps: int, learning_repeat_times: int = 20):
        # env running steps
        env_running_step = 0
        obs = self.env.reset().copy()
        evaluate_model_flag = False
        while self.learning_step < total_learning_steps:
            action = self.actor.act(obs)
            next_obs, reward, done, _ = self.env.step(action)
            env_running_step += 1
            transition = {
                "observation": np.array(obs[None, :, :, :], dtype=np.float32),
                "action": np.array(action[None, :], np.float32),
                "reward": np.array([reward], dtype=np.float32),
                "next_observation": np.array(next_obs[None, :, :, :], dtype=np.float32),
                "done": np.array([done], dtype=np.bool_),
            }
            self.reply_buffer.store(transition)
            # update observation
            if done:
                if evaluate_model_flag:
                    self.evaluate_model()
                    evaluate_model_flag = False
                obs = self.env.reset().copy()
            else:
                obs = next_obs

            # learning a model
            if env_running_step % self.learning_iter == 0:
                for _ in range(learning_repeat_times):
                    self.learn_model()
                    self.learning_step += 1
                    if self.learning_step % self.evaluate_iter == 0:
                        evaluate_model_flag = True

    def learn_model(self):
        # Setting train mode
        self.actor.train()
        self.critic_v.train()
        self.critic_q.train()

        sample_transitions = self.reply_buffer.sample(self.sample_batch_size)
        for key in sample_transitions.keys():
            sample_transitions[key] = torch.tensor(
                sample_transitions[key], device=self.training_device
            )
        # SAC update algorithm
        # update critic v
        obs = sample_transitions["observation"]
        with torch.no_grad():
            _, sample_actions, actions_log_prob = self.actor.sample(obs)
            q_value_of_obs_sample_action = self.critic_q(obs, sample_actions)
            v_value_target = (
                q_value_of_obs_sample_action - self.temperature_factor * actions_log_prob
            )
        v_value_of_obs = self.critic_v(obs)
        v_value_loss = self.loss_function(v_value_of_obs, v_value_target)
        self.v_value_optimizer.zero_grad()
        v_value_loss.backward()
        self.v_value_optimizer.step()
        # recording the v value loss
        self.writer.add_scalar(
            "SAC/v_value_loss", v_value_loss.cpu(), self.learning_step
        )

        # update critic q
        actions = sample_transitions["action"]
        rewards = sample_transitions["reward"]
        next_obs = sample_transitions["next_observation"]
        q_value_of_obs_action = self.critic_q(obs, actions)
        with torch.no_grad():
            q_value_target = rewards + self.gamma * self.target_critic_v(next_obs)
        q_value_loss = self.loss_function(q_value_of_obs_action, q_value_target)
        self.q_value_optimizer.zero_grad()
        q_value_loss.backward()
        self.q_value_optimizer.step()
        # recording the q value loss
        self.writer.add_scalar(
            "SAC/q_value_loss", q_value_loss.cpu(), self.learning_step
        )

        # update policy pi
        _, sample_actions, actions_log_prob = self.actor.sample(obs)
        q_value_of_obs_sample_action = self.critic_q(obs, sample_actions)
        actor_loss = torch.mean(
            self.temperature_factor * actions_log_prob - q_value_of_obs_sample_action
        )
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # recording the actor loss
        self.writer.add_scalar("SAC/actor_loss", actor_loss.cpu(), self.learning_step)

        # soft update target v value network
        for target_critic_v_param, critic_v_param in zip(
            self.target_critic_v.parameters(), self.critic_v.parameters()
        ):
            target_critic_v_param.data.copy_(
                self.tau * critic_v_param + (1 - self.tau) * target_critic_v_param
            )

    def evaluate_model(self, total_evaluating_steps: int = 100) -> None:
        self.env.is_test = True
        evaluating_step = 0
        successful_times = 0

        obs = self.env.reset().copy()
        while evaluating_step < total_evaluating_steps:
            action = self.actor.act(obs, evaluation_mode=True)
            next_obs, reward, done, _ = self.env.step(action)
            if done:
                obs = self.env.reset().copy()
                evaluating_step += 1
                if reward > 0.9:
                    successful_times += 1
            else:
                obs = next_obs

        self.writer.add_scalar(
            "SAC/actor_successful_rate",
            np.array(successful_times / total_evaluating_steps),
            self.learning_step,
        )

        self.env.is_test = False

    def save_model(self):
        model = {
            "actor": self.actor,
            "critic_q": self.critic_q,
            "critic_v": self.critic_v,
            "target_critic_v": self.target_critic_v,
            "learning_step": self.learning_step,
        }
        torch.save(model, self.model_file_path)

    def load_model(self):
        model = torch.load(self.model_file_path, map_location=self.training_device)
        self.actor = model["actor"]
        self.critic_q = model["critic_q"]
        self.critic_v = model["critic_v"]
        self.target_critic_v = model["target_critic_v"]
        self.learning_step = model["learning_step"]
