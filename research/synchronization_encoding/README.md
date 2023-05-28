# synchronization encoding
A simple experiment for exploring the synchronization encoding of neural networks, which may be used to reinforcement learning.

## envs
Continuous Mountain Car

## ideas
本次想法就是，通过一个带有dropout层（或者不带dropout层，使用高斯噪声）的神经网络构建一个actor，这样网络自带探索性。如果在获得一次obs后得到了一次action，并作用于环境，得到奖励。如果奖励越大，那么对于该action整体使用贪心算法，让网络整体的兴奋度更高。就比如，该网络可以写为$\pi_{\theta}$。内部包含两层隐藏层$a = \pi_{\theta}(s)$。网络的兴奋度函数为两层隐藏层其激活后发放的值的square（或者其他值），如果经过一个action后，得到了一个奖励$r$, 那么改奖励就比如与整个神经网络的兴奋度成比例关系或者是 $log$ 关系。

## problems
- 如何使得神经网络的兴奋性与奖励函数成正比
