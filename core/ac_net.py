import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .img_stack import ImgStack
from .state_stack import StateStack
from .init_weight import initWeights

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-8


class Policy(nn.Module):
    """动作策略
    输出一个最优动作或者一个动作概率分布，默认高斯分布

    """

    def __init__(
        self,
        imageSize=[1, 160, 160],
        action_dim=5,
    ):
        super(Policy, self).__init__()

        self.imgStack = ImgStack(input_size=imageSize)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1)

        self.linear1 = nn.Linear(16 * 24 * 24, 128)
        # 输出动作
        self.mean_action_linear = nn.Linear(128, action_dim)
        self.log_std_linear = nn.Linear(128, action_dim)

        self.apply(initWeights)

        # 动作放缩
        self.action_scale = torch.tensor(1.0)
        self.action_bias = torch.tensor(0.0)

    def forward(self, img):
        self.batch_size = img.shape[0]
        x = self.imgStack(img)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((self.batch_size, -1))
        x = F.relu(self.linear1(x))
        # size (batch_size, 5)
        mean = self.mean_action_linear(x).squeeze(-1)
        log_std = self.log_std_linear(x).squeeze(-1)
        # 返回动作
        return mean, log_std

    def sample(self, img):
        mean, log_std = self.forward(img)
        std = log_std.exp()
        # 构建高斯分布
        # mean + std * N(0,1)
        normal = Normal(mean, std)
        # 重新参数化，实现可微
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        # 获取action
        action = y_t * self.action_scale + self.action_bias
        # 获取熵
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        # 获取动作均值
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean


class QNet(nn.Module):
    """Critic"""

    def __init__(
        self,
        imageSize=[1, 128, 128],
        stateSize=[
            5,
        ],
    ):
        super(QNet, self).__init__()

        self.imgStack = ImgStack(input_size=imageSize)
        self.stateStack = StateStack(input_size=stateSize)

        self.conv1 = nn.Conv2d(33, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1)

        self.q_1_linear1 = nn.Linear(16 * 24 * 24, 128)
        self.q_1_linear2 = nn.Linear(128, 64)
        # 输出q
        self.q_1_values = nn.Linear(64, 1)

        self.q_2_linear1 = nn.Linear(16 * 24 * 24, 128)
        self.q_2_linear2 = nn.Linear(128, 64)
        # 输出q
        self.q_2_values = nn.Linear(64, 1)

        self.apply(initWeights)

    def forward(self, img, state):
        self.batch_size = img.shape[0]

        img = self.imgStack(img)
        state = self.stateStack(state)
        x = torch.cat((img, state), dim=1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view((self.batch_size, -1))
        x_1 = F.relu(self.q_1_linear1(x))
        x_2 = F.relu(self.q_2_linear1(x))
        x_1 = F.relu(self.q_1_linear2(x_1))
        x_2 = F.relu(self.q_2_linear2(x_2))
        # state_values
        # size (batch_size, 1)
        q_1 = self.q_1_values(x_1)
        q_2 = self.q_2_values(x_2)

        # 返回评价
        return q_1, q_2
