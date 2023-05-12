import os
from typing import Callable, Tuple

import numpy as np

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from ...memory import ReplyBuffer
from .actor import Actor
from .critic import CriticQ, CriticV


class SAC:
    def __init__(
        self,
        env: Callable,
        buffer_capacity: int = 2000,
        gamma: float = 0.98,
        temperature_factor: float = 0.8,
        lr: float = [1e-5, 1e-5, 1e-5],
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
        self.lr_actor = lr[0]
        self.lr_critic_v = lr[1]
        self.lr_critic_q = lr[2]
        self.tau = 1e-3
        # gradient clip
        self.grad_max_norm = 1

        # some frequency
        self.net_analysis_iter = 1000
        self.learning_iter = 40
        self.evaluate_iter = 2000

        # buffer setting
        self.reply_buffer = ReplyBuffer(capacity=buffer_capacity)
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
            self.critic_q1 = CriticQ(
                self.obs_dim, self.action_dim, device=self.training_device
            )
            self.critic_q2 = CriticQ(
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
        # Use Adam
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.lr_actor)
        self.v_value_optimizer = torch.optim.Adam(
            self.critic_v.parameters(), self.lr_critic_v
        )
        self.q1_value_optimizer = torch.optim.Adam(
            self.critic_q1.parameters(), self.lr_critic_q
        )
        self.q2_value_optimizer = torch.optim.Adam(
            self.critic_q2.parameters(), self.lr_critic_q
        )

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
        self.critic_q1.train()
        self.critic_q2.train()

        sample_transitions = self.reply_buffer.sample(self.sample_batch_size)
        for key in sample_transitions.keys():
            sample_transitions[key] = torch.tensor(
                sample_transitions[key], device=self.training_device
            )
        # SAC update algorithm
        # update critic v
        # calculate v loss
        obs = sample_transitions["observation"]
        _, sample_actions, actions_log_prob = self.actor.sample(obs)
        q_value_of_obs_sample_action = torch.min(
            self.critic_q1(obs, sample_actions), self.critic_q2(obs, sample_actions)
        )
        v_value_target = (
            q_value_of_obs_sample_action - self.temperature_factor * actions_log_prob
        )
        v_value_of_obs = self.critic_v(obs)
        v_value_loss = self.loss_function(v_value_of_obs, v_value_target)
        # update v function parameters
        self.v_value_optimizer.zero_grad()
        v_value_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic_v.parameters(), self.grad_max_norm)
        self.v_value_optimizer.step()
        # recording the v value loss
        self.writer.add_scalar(
            "SAC/v_value_loss", v_value_loss.cpu(), self.learning_step
        )

        # update policy pi
        # calculate policy loss
        actor_loss = torch.mean(
            self.temperature_factor * actions_log_prob - q_value_of_obs_sample_action
        )
        # update policy parameters
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor.parameters(), self.grad_max_norm)
        self.actor_optimizer.step()
        # recording the actor loss
        self.writer.add_scalar("SAC/actor_loss", actor_loss.cpu(), self.learning_step)

        # update critic q
        # calculate q1 and q2 value loss
        actions = sample_transitions["action"]
        rewards = sample_transitions["reward"]
        next_obs = sample_transitions["next_observation"]
        q_value_target = rewards + self.gamma * self.target_critic_v(next_obs)
        # q1 value loss
        q1_value_of_obs_action = self.critic_q1(obs, actions)
        q1_value_loss = self.loss_function(q1_value_of_obs_action, q_value_target)
        # update q1 function parameters
        self.q1_value_optimizer.zero_grad()
        q1_value_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic_q1.parameters(), self.grad_max_norm)
        self.q1_value_optimizer.step()
        # q2 value loss
        q2_value_of_obs_action = self.critic_q2(obs, actions)
        q2_value_loss = self.loss_function(q2_value_of_obs_action, q_value_target)
        # update q2 function parameters
        self.q2_value_optimizer.zero_grad()
        q2_value_loss.backward()
        clip_grad_norm_(self.critic_q2.parameters(), self.grad_max_norm)
        self.q2_value_optimizer.step()
        # recording the q1 and q2 value loss
        self.writer.add_scalar(
            "SAC/q1_value_loss", q1_value_loss.cpu(), self.learning_step
        )
        self.writer.add_scalar(
            "SAC/q2_value_loss", q2_value_loss.cpu(), self.learning_step
        )

        # soft update target v value network
        for target_critic_v_param, critic_v_param in zip(
            self.target_critic_v.parameters(), self.critic_v.parameters()
        ):
            target_critic_v_param.data.copy_(
                self.tau * critic_v_param + (1 - self.tau) * target_critic_v_param
            )

    def evaluate_model(self, total_evaluating_steps: int = 100) -> None:
        self.env.is_test = True
        self.actor.eval()

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
            "critic_v": self.critic_v,
            "critic_q1": self.critic_q1,
            "critic_q2": self.critic_q2,
            "target_critic_v": self.target_critic_v,
            "learning_step": self.learning_step,
        }
        torch.save(model, self.model_file_path)

    def load_model(self):
        model = torch.load(self.model_file_path, map_location=self.training_device)
        self.actor = model["actor"]
        self.critic_v = model["critic_v"]
        self.critic_q1 = model["critic_q1"]
        self.critic_q2 = model["critic_q2"]
        self.target_critic_v = model["target_critic_v"]
        self.learning_step = model["learning_step"]
