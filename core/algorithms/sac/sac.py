import os
from typing import Callable, Tuple

import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

from ...memory import ReplayBuffer

from .actor import Actor
from .critic import CriticQ


class SAC:
    """Soft Actor Critic algorithm."""

    def __init__(
        self,
        env: Callable,
        buffer_capacity: int = 2000,
        gamma: float = 0.99,
        tau: float = 5e-3,
        action_scale: float = 1.0,
        lr: Tuple[float, float] = (3e-4, 3e-4, 3e-4),
        optimizer: str = "Adam",
        logfile_dir: str = "log/",
    ) -> None:
        """Initialize the model parameters.

        Args:
            env: the MDP or POMDP process
            buffer_capacity: the max size of the buffer
            gamma: the reward discount coefficient

        """
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
        self.tau = tau
        self.action_scale = action_scale
        self.lr_actor = lr[0]
        self.lr_critic_q = lr[1]
        self.lr_log_alpha = lr[2]
        optimizer_dict = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD}
        self.optimizer = optimizer_dict[optimizer]
        # gradient clip
        self.max_grad_norm = 1

        # buffer setting
        self.reply_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.minimal_buffer_length = 1000

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
        self.model_save_dir = logfile_dir + env.env_name + "/sac/model/"
        self.model_file_path = logfile_dir + env.env_name + "/sac/model/sac_model.pkl"
        self.train_logfile_dir = logfile_dir + env.env_name + "/sac/train_log/"
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
            self.critic_q1 = CriticQ(
                self.obs_dim, self.action_dim, device=self.training_device
            )
            self.critic_q2 = CriticQ(
                self.obs_dim, self.action_dim, device=self.training_device
            )
            self.target_critic_q1 = CriticQ(
                self.obs_dim, self.action_dim, device=self.training_device
            )
            self.target_critic_q2 = CriticQ(
                self.obs_dim, self.action_dim, device=self.training_device
            )
            self.actor = Actor(
                self.obs_dim,
                self.action_dim,
                action_scale=self.action_scale,
                device=self.training_device,
            )
            self.log_alpha = torch.tensor(
                [np.log(0.01)],
                device=self.training_device,
                dtype=torch.float,
                requires_grad=True,
            )
            # Initialize the target q1 and q2 net
            self.target_critic_q1.load_state_dict(self.critic_q1.state_dict())
            self.target_critic_q2.load_state_dict(self.critic_q2.state_dict())
            # training step, support checkpoint
            self.learning_step = 0
        # Initialize the optimizer
        self.actor_optimizer = self.optimizer(self.actor.parameters(), self.lr_actor)
        self.q1_value_optimizer = self.optimizer(
            self.critic_q1.parameters(), self.lr_critic_q
        )
        self.q2_value_optimizer = self.optimizer(
            self.critic_q2.parameters(), self.lr_critic_q
        )
        self.log_alpha_optimizer = self.optimizer([self.log_alpha], self.lr_log_alpha)

        # Loss function
        self.loss_function = nn.MSELoss()

    def learn(
        self,
        max_learning_steps: int,
        sample_batch_size: int = 64,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
        learning_period: int = 1,
        evaluating_period: int = 1000,
        evaluating_times: int = 1,
    ) -> None:
        env_running_step = 0
        evaluating_model_flag = False
        obs = self.env.reset()
        while self.learning_step < max_learning_steps:
            action = self.actor.act(obs)
            next_obs, reward, done, _ = self.env.step(action)
            env_running_step += 1
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
            self.reply_buffer.store(transition)
            # update observation
            if done:
                obs = self.env.reset()
                if evaluating_model_flag:
                    self.evaluate_model(evaluating_times, record=True)
                    evaluating_model_flag = False
            else:
                obs = next_obs
            # learning a model
            if (
                self.reply_buffer.length >= self.minimal_buffer_length
                and env_running_step % learning_period == 0
            ):
                self.learn_model(sample_batch_size)
                self.learning_step += 1
                if self.learning_step % evaluating_period == 0:
                    evaluating_model_flag = True

    def learn_model(self, batch_size: int) -> None:
        # Setting train mode
        self.actor.train()
        self.critic_q1.train()
        self.critic_q2.train()

        sample_transitions = self.reply_buffer.sample(batch_size)
        for key in sample_transitions.keys():
            sample_transitions[key] = torch.tensor(
                sample_transitions[key], device=self.training_device
            )
        # SAC update algorithm
        obs = sample_transitions["observation"]
        actions = sample_transitions["action"]
        rewards = sample_transitions["reward"]
        next_obs = sample_transitions["next_observation"]
        done = sample_transitions["done"]
        with torch.no_grad():
            q_target_value = self.calculate_q_target(next_obs, rewards, done)
        # q value loss
        q1_value_loss = self.loss_function(self.critic_q1(obs, actions), q_target_value)
        q2_value_loss = self.loss_function(self.critic_q2(obs, actions), q_target_value)
        # update q1 function parameters
        self.q1_value_optimizer.zero_grad()
        q1_value_loss.backward()
        self.q1_value_optimizer.step()
        # update q2 function parameters
        self.q2_value_optimizer.zero_grad()
        q2_value_loss.backward()
        self.q2_value_optimizer.step()
        # recording the q1 and q2 value loss
        self.writer.add_scalar(
            "SAC/q1_value_loss", q1_value_loss.cpu(), self.learning_step
        )
        self.writer.add_scalar(
            "SAC/q2_value_loss", q2_value_loss.cpu(), self.learning_step
        )
        # update policy pi
        # calculate policy loss
        _, new_sample_actions, actions_log_prob = self.actor.sample(obs)
        new_q1_value = self.critic_q1(obs, new_sample_actions)
        new_q2_value = self.critic_q2(obs, new_sample_actions)
        actor_loss = torch.mean(
            self.log_alpha.exp() * actions_log_prob
            - torch.min(new_q1_value, new_q2_value)
        )
        # update policy parameters
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # recording the actor loss
        self.writer.add_scalar("SAC/actor_loss", actor_loss.cpu(), self.learning_step)

        # update temperature factor
        entropy = -actions_log_prob
        log_alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp()
        )
        self.log_alpha_optimizer.zero_grad()
        log_alpha_loss.backward()
        self.log_alpha_optimizer.step()
        # recording the temperature factor
        alpha_detached = torch.exp(self.log_alpha.detach())
        self.writer.add_scalar(
            "SAC/temperature_factor", alpha_detached.cpu(), self.learning_step
        )

        # soft update target v value network
        self.soft_update(self.target_critic_q1, self.critic_q1)
        self.soft_update(self.target_critic_q2, self.critic_q2)

    def evaluate_model(
        self, total_evaluating_episodes: int = 100, record: bool = False
    ) -> None:
        self.env.is_test = True
        self.actor.eval()

        evaluating_step = 0
        sum_reward = 0

        obs = self.env.reset()
        while evaluating_step < total_evaluating_episodes:
            action = self.actor.act(obs, evaluation_mode=True)
            next_obs, reward, done, _ = self.env.step(action)
            sum_reward += reward
            if done:
                obs = self.env.reset()
                evaluating_step += 1
            else:
                obs = next_obs
        if record:
            self.writer.add_scalar(
                "SAC/mean_reward",
                np.array(sum_reward / total_evaluating_episodes),
                self.learning_step,
            )
        else:
            print(f"The mean return reward is {sum_reward / total_evaluating_episodes}")
        self.env.is_test = False

    def calculate_q_target(
        self, next_obs: Tensor, reward: Tensor, done: Tensor
    ) -> Tensor:
        _, next_sample_action, next_action_log_prob = self.actor.sample(next_obs)
        q1_value = self.target_critic_q1(next_obs, next_sample_action)
        q2_value = self.target_critic_q2(next_obs, next_sample_action)
        next_v_value = (
            torch.min(q1_value, q2_value) - self.log_alpha.exp() * next_action_log_prob
        )

        return reward + self.gamma * next_v_value * (1 - done)

    def soft_update(self, target_net: Module, net: Module) -> None:
        for target_net_param, net_param in zip(
            target_net.parameters(), net.parameters()
        ):
            target_net_param.data.copy_(
                self.tau * net_param.data + (1 - self.tau) * target_net_param.data
            )

    def save_model(self):
        model = {
            "actor": self.actor,
            "critic_q1": self.critic_q1,
            "critic_q2": self.critic_q2,
            "target_critic_q1": self.target_critic_q1,
            "target_critic_q2": self.target_critic_q2,
            "log_alpha": self.log_alpha,
            "learning_step": self.learning_step,
        }
        torch.save(model, self.model_file_path)

    def load_model(self):
        model = torch.load(self.model_file_path, map_location=self.training_device)
        self.actor = model["actor"]
        self.critic_q1 = model["critic_q1"]
        self.critic_q2 = model["critic_q2"]
        self.target_critic_q1 = model["target_critic_q1"]
        self.target_critic_q2 = model["target_critic_q2"]
        self.log_alpha = model["log_alpha"]
        self.learning_step = model["learning_step"]
