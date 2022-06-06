"""
@author Bruce Hou
@brief 
@version 0.1
@date 2022-01-14

@copyright Copyright (c) 2022
"""
import math
from tkinter import Place
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal  # << 多元高斯噪声
from .ACNet import Policy, QNet


class CoreSACPolicy(nn.Module):
    """
    @brief Actor-Critic 
    """

    def __init__(
        self, imageSize=[1, 128, 128], action_dim=5,
    ):
        super(CoreSACPolicy, self).__init__()

        self.action_dim = action_dim  # << 动作维度
        # 动作选择策略
        self.policy = Policy(imageSize, self.action_dim)

    def selectAction(self, img, evaluate=False):
        """
        @brief 探索
        """
        if torch.cuda.is_available():
            img = torch.FloatTensor(torch.from_numpy(img)).cuda()
        else:
            img = torch.FloatTensor(torch.from_numpy(img))

        img = img.unsqueeze(0)

        with torch.no_grad():
            if evaluate is False:
                action, _, _ = self.policy.sample(img)
            else:
                _, _, action = self.policy.sample(img)

        return action.squeeze().cpu().numpy()


class CoreSACQNet(nn.Module):
    def __init__(
        self, imageSize=[1, 128, 128], action_dim=5,
    ):
        super(CoreSACQNet, self).__init__()
        # 两个q函数
        self.q_net = QNet(imageSize, stateSize=[action_dim,])

if __name__ == "__main__":
    pass
