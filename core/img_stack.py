import torch
import torch.nn as nn
import torch.nn.functional as F


class ImgStack(nn.Module):
    def __init__(self, input_size=[1, 128, 128]):
        super(ImgStack, self).__init__()

        # 第一个卷积层
        self.conv1 = nn.Conv2d(input_size[0], 128, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # 使用池化层降低计算量
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二个卷积层
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool2(x)

        return x


if __name__ == "__main__":
    # GPU计时器
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        # User code
        tensor = torch.randn((16, 1, 128, 128)).cuda()
        img_stack = ImgStack().cuda()
        print(img_stack)
        # img_stack = img_stack.cuda()
        result = img_stack.forward(tensor)
        print(result.shape)
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        # 打印时间
        print("运行时间：", start.elapsed_time(end), "ms")

