import torch
from thunder.nn import AdaptReducingConvBlock, MlpBlock, ReducingConvBlock

# test mlp block
mlp = MlpBlock((10, 256, 256, 5), activation_fn="relu")
print(f"The MLP Model is {mlp}")
x = torch.randn(2, 10)
y = mlp(x)
print(f"The MLP Model's output is {y}")

# test reducing convolution block
reducing_conv_block = ReducingConvBlock(
    3, 256, activation_fn="softsign", activate_output=True
)
print(f"The convolution block architecture is {reducing_conv_block}")
x = torch.rand(2, 3, 256, 256)
y = reducing_conv_block(x)
print(f"The feature map after the reducing convolution block is {y}")

# test adaptive reducing convolution block
adapt_reducing_conv_block = AdaptReducingConvBlock(
    in_channels=3, activation_fn="softsign", activate_output=True
)
print(
    f"The adaptive reducing convolution block architecture is {adapt_reducing_conv_block}"
)
