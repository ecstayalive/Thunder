"""
@author Bruce Hou
@brief 
@version 1.0
@date 2022-01-14
       
@copyright Copyright (c) 2022
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from .Core import CoreSACPolicy, CoreSACQNet
from .Buffer import Buffer

import numpy as np


class GPModel:
    def __init__(
        self,
        env,
        total_timesteps,
        epochs=20,
        buffer_size=10000,
        lr_policy=1e-4,
        lr_q=1e-4,
        lr_alpha=1e-4,
        gamma=0.9,
        batch_size=16,
        file_path="Model/",
        tensorboard_log="Log/",
    ):

        # 环境配置
        self.env = env
        self.action_dim = env.action_space.shape[0]
        self.img_size = env.observation_space.shape
        self.maxStep = env._maxSteps  # << 环境设定步数，即在这些步中完成任务
        self.pre_train_step = 2000

        # 训练网络的一些配置参数
        self.total_timesteps = total_timesteps  # << 每一个episode总的训练步数
        self.epochs = epochs

        # 网络参数配置
        self.lr_policy = lr_policy  # << 学习率
        self.lr_q = lr_q  # << 学习率
        self.clip_grad = 1  # << 梯度裁剪，防止梯度消失
        self.tau = 0.001  # << 指数更新参数
        self.gamma = gamma  # << gamma
        self.alpha = 1  # << 温度系数

        # 频率配置
        self.net_analysis_iter = 1000  # << 网络可视化频率
        self.learning_iter = 40  # 每收集learning_iter个数据学习epochs次
        self.evaluate_iter = 1000  # 每学习evaluate_iter个数据评估一次
        self.decay_action_var_iter = 5000

        # memory配置
        self.capacity = buffer_size  # << 经验池容量
        self.batch_size = batch_size

        self.file_path = file_path
        self.tensorboard_log = tensorboard_log

        # 一些配置
        self.grasp_times = 0  # << 抓取次数
        self.success_rate = 0  # << 成功率

        # 记录数据
        self.writer = SummaryWriter(log_dir=tensorboard_log)

        # 判断是否有gpu
        if os.path.exists(self.file_path + "model.pkl"):
            print("存在模型数据，加载模型数据")
            self.load()
        else:
            print("建立模型")
            if torch.cuda.is_available():
                self.eval_net, self.target_net = (
                    CoreSACQNet(
                        imageSize=self.img_size, action_dim=self.action_dim
                    ).cuda(),
                    CoreSACQNet(
                        imageSize=self.img_size, action_dim=self.action_dim
                    ).cuda(),
                )
                self.policy_net = CoreSACPolicy(
                    imageSize=self.img_size, action_dim=self.action_dim
                ).cuda()
                self.target_entropy = (
                    torch.log(torch.Tensor((self.action_dim,))).cuda().item()
                )
                self.log_alpha = torch.zeros(1, requires_grad=True, device="cuda")
            else:
                self.eval_net, self.target_net = (
                    CoreSACQNet(imageSize=self.img_size, action_dim=self.action_dim),
                    CoreSACQNet(imageSize=self.img_size, action_dim=self.action_dim),
                )
                self.policy_net = CoreSACPolicy(
                    imageSize=self.img_size, action_dim=self.action_dim
                )
                self.target_entropy = torch.log(
                    torch.Tensor((self.action_dim,))
                ).item()
                self.log_alpha = torch.zeros(1, requires_grad=True)
            # 初始化网络
            for target_param, eval_net in zip(
                self.target_net.parameters(), self.eval_net.parameters(),
            ):
                target_param.data.copy_(eval_net.data)
            self.env_step = 0  # << 环境计步
            self.learning_step = 0  # << 学习计步

            # 优化器
            self.policy_optimizer = torch.optim.Adam(
                self.policy_net.policy.parameters(), lr_policy
            )
            self.q_net_optimizer = torch.optim.Adam(
                self.eval_net.q_net.parameters(), lr_q
            )
            # 优化器
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        if torch.cuda.is_available():
            self.loss = nn.MSELoss().cuda()
        else:
            self.loss = nn.MSELoss()

    def learn(self,):
        """
        @brief 与环境交互，学习模型
        """
        self.buffer = Buffer(
            capacity=self.capacity,
            batch_size=self.batch_size,
            action_dim=self.action_dim,
        )
        obs = self.env.reset().copy()

        collect_step = 0
        while self.env_step <= self.total_timesteps:
            # 收集数据
            action = self.policy_net.selectAction(img=obs)
            next_obs, reward, done, _ = self.env.step(action)
            self.buffer.store(obs, action, reward, next_obs, done)
            # 步数累加
            # 更新状态
            if done:
                obs = self.env.reset().copy()
                self.grasp_times += 1
                collect_step = 0
            else:
                obs = next_obs
            # 更新计步
            collect_step += 1
            self.env_step += 1

            # 训练
            if self.env_step % self.learning_iter == 0:
                for _ in range(self.epochs):
                    self.train()

            # 评估表现
            if self.grasp_times % self.evaluate_iter == 0 and self.grasp_times > 0:
                self.evaluate()
                self.grasp_times += 1

    def evaluate(self, evaluate_times=100, record=True):
        """
        @brief 评估模型
        """
        self.env._isTest = True
        obs = self.env.reset().copy()
        eval_step = 0
        sum_reward = 0
        step = 0
        while eval_step < evaluate_times:
            action = self.policy_net.selectAction(img=obs, evaluate=True)
            next_obs, reward, done, _ = self.env.step(action)
            step += 1
            sum_reward += reward
            # 更新状态
            if done:
                obs = self.env.reset().copy()
                eval_step += 1
                step = 0
                print(
                    f"正确率: ", {sum_reward / evaluate_times}, ",  已测试步数: ", {eval_step},
                )
            else:
                obs = next_obs
        success_rate = sum_reward / evaluate_times
        if record:
            if success_rate >= self.success_rate:
                # 保存模型
                self.save()
                self.success_rate = success_rate

            self.writer.add_scalar(
                "NET/evaluation", np.array(success_rate), self.learning_step
            )
        self.env._isTest = False

    def train(self):
        """
        @brief 训练神经网络
        """
        self.eval_net.train()
        # 抽取Memory中的batch_size个数据
        data_batch = self.buffer.sample()
        if torch.cuda.is_available():
            img = torch.FloatTensor(torch.from_numpy(data_batch[0])).cuda()
            a = torch.FloatTensor(torch.from_numpy(data_batch[1])).cuda()
            r = torch.FloatTensor(torch.from_numpy(data_batch[2])).cuda()
            img_ = torch.FloatTensor(torch.from_numpy(data_batch[3])).cuda()
            d = torch.ByteTensor(torch.from_numpy(data_batch[4])).cuda()
        else:
            img = torch.FloatTensor(torch.from_numpy(data_batch[0]))
            a = torch.FloatTensor(torch.from_numpy(data_batch[1]))
            r = torch.FloatTensor(torch.from_numpy(data_batch[2]))
            img_ = torch.FloatTensor(torch.from_numpy(data_batch[3]))
            d = torch.ByteTensor(torch.from_numpy(data_batch[4]))

        # 计算q_target
        with torch.no_grad():
            next_action, next_log_pi, _ = self.policy_net.policy.sample(img_)
            q_1_next_target, q_2_next_target = self.eval_net.q_net(img_, next_action)
            # 计算target
            min_qf_next_target = (
                torch.min(q_1_next_target, q_2_next_target) - self.alpha * next_log_pi
            )
            # (self.batch_size, 1)
            q_target = (
                r.unsqueeze(-1)
                + (1 - d).unsqueeze(-1) * self.gamma * min_qf_next_target
            )
        ########################################################################
        # train QNet
        ########################################################################
        q1, q2 = self.eval_net.q_net(img, a)
        q1_loss = self.loss(q1, q_target)
        q2_loss = self.loss(q2, q_target)
        q_loss = q1_loss + q2_loss
        # 记录q和q_loss数据
        q_eval_value = torch.min(q1.detach(), q2.detach())
        q1_loss_value = q1_loss.detach()
        q2_loss_value = q2_loss.detach()
        # 可视化q值
        self.writer.add_scalar(
            "NET/q_eval_value", torch.mean(q_eval_value.cpu()), self.learning_step
        )
        self.writer.add_scalar(
            "NET/q1_loss_value", torch.mean(q1_loss_value.cpu()), self.learning_step
        )
        self.writer.add_scalar(
            "NET/q2_loss_value", torch.mean(q2_loss_value.cpu()), self.learning_step
        )
        # 梯度下降，使得q值收敛
        # 清空梯度
        self.q_net_optimizer.zero_grad()
        q_loss.backward()
        clip_grad_norm_(self.eval_net.q_net.parameters(), self.clip_grad)
        self.q_net_optimizer.step()
        ########################################################################
        # 更新动作选择网络
        ########################################################################
        pi, log_pi, _ = self.policy_net.policy.sample(img)
        qf1_pi, qf2_pi = self.eval_net.q_net(img, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        # 动作选择策略的损失
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        policy_loss_value = policy_loss.detach()
        # 可视化动作选择策略的损失
        self.writer.add_scalar(
            "NET/policy_loss_value",
            torch.mean(policy_loss_value.cpu()),
            self.learning_step,
        )
        #
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        clip_grad_norm_(self.policy_net.policy.parameters(), self.clip_grad)
        self.policy_optimizer.step()
        ########################################################################
        # 更新温度系数
        ########################################################################
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        alpha_loss_value = alpha_loss.detach()
        # 可视化熵损失
        self.writer.add_scalar(
            "NET/entropy_loss", alpha_loss_value.cpu(), self.learning_step,
        )

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        clip_grad_norm_(self.log_alpha, self.clip_grad)
        self.alpha_optimizer.step()
        # 更新温度系数
        self.alpha = self.log_alpha.exp()
        alpha_tlogs = self.alpha.clone()
        # 可视化温度系数
        self.writer.add_scalar(
            "NET/alpha_value", torch.mean(alpha_tlogs.cpu()), self.learning_step,
        )
        # 步数加1
        self.learning_step += 1

        # 分析网络模型
        if self.learning_step % self.net_analysis_iter == 0:
            self.analyzeNet()

        # 指数滞后版本
        for target_param, eval_param in zip(
            self.target_net.parameters(), self.eval_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * eval_param.data + (1 - self.tau) * target_param.data
            )

    def analyzeNet(self,):
        """
        @brief 分析神经网络中的权重，参数和卷积层可视化
        """
        with torch.no_grad():
            for name, param in self.eval_net.named_parameters():
                if param.grad is not None:
                    self.writer.add_histogram(
                        "Model/" + name + "_param", param, self.learning_step,
                    )
                    self.writer.add_histogram(
                        "Model/" + name + "_grad", param.grad, self.learning_step,
                    )
                else:
                    pass
            for name, param in self.policy_net.named_parameters():
                if param.grad is not None:
                    self.writer.add_histogram(
                        "Model/" + name + "_param", param, self.learning_step,
                    )
                    self.writer.add_histogram(
                        "Model/" + name + "_grad", param.grad, self.learning_step,
                    )
                else:
                    pass

    def save(self):
        """
        @brief 保存模型
        """
        model = {
            "eval_net": self.eval_net,
            "target_net": self.target_net,  # << 指数滞后版本
            "policy_net": self.policy_net,
            "policy_optimizer": self.policy_optimizer,
            "q_net_optimizer": self.q_net_optimizer,
            "alpha_optimizer": self.alpha_optimizer,
            "target_entropy": self.target_entropy,
            "log_alpha": self.log_alpha,
            "env_step": self.env_step,
            "learning_step": self.learning_step,
            "success_rate": self.success_rate,
        }
        print("\n保存模型")
        torch.save(model, self.file_path + "model.pkl")
        self.buffer.save()
        print("保存完成")

    def load(self):
        """
        @brief 加载模型
        """
        if torch.cuda.is_available():
            model = torch.load(self.file_path + "model.pkl", map_location="cuda")
            self.eval_net = model["eval_net"]
            self.target_net = model["target_net"]
            self.policy_net = model["policy_net"]
            self.policy_optimizer = model["policy_optimizer"]
            self.q_net_optimizer = model["q_net_optimizer"]
            self.alpha_optimizer = model["alpha_optimizer"]
            self.log_alpha = model["log_alpha"]
            self.target_entropy = model["target_entropy"]
            self.env_step = model["env_step"]
            self.learning_step = model["learning_step"]
            self.success_rate = model["success_rate"]
        else:
            model = torch.load(self.file_path + "model.pkl", map_location="cpu")
            self.eval_net = model["eval_net"]
            self.target_net = model["target_net"]
            self.policy_net = model["policy_net"]
            self.policy_optimizer = model["policy_optimizer"]
            self.q_net_optimizer = model["q_net_optimizer"]
            self.alpha_optimizer = model["alpha_optimizer"]
            self.log_alpha = model["log_alpha"]
            self.target_entropy = model["target_entropy"]
            self.env_step = model["env_step"]
            self.learning_step = model["learning_step"]
            self.success_rate = model["success_rate"]
