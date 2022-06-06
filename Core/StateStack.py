"""
@author Bruce Hou
@brief 
@version 0.1
@date 2022-01-14

@copyright Copyright (c) 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateStack(nn.Module):
    """
    @brief 当网络需要输入action以及gripper的高度或者其是否关闭时使用
    """

    def __init__(self, input_size=[6,], output_size=[28 * 28,]):
        """
        @param input_size: 刚开始时，输入action输入三轴动作增量，同时输入gripper的角度，开关信号，
                           以及额外状态gripper的高度，一共六个量
        """
        super(StateStack, self).__init__()

        # 第一个卷积层
        self.linear1 = nn.Linear(input_size[0], 128)
        self.linear2 = nn.Linear(128, output_size[0])

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # 转化为四维张量
        x = x.view(-1, 28, 28).unsqueeze(1)
        return x


if __name__ == "__main__":
    # GPU计时器
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        # User code
        tensor = torch.randn((16, 6)).cuda()
        state_stack = StateStack().cuda()
        state_stack.forward(tensor)
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
        # 打印时间
        print("运行时间：", start.elapsed_time(end), "ms")

