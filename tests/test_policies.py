import torch
from thunder.policies import CnnGaussianPolicy, GeneralGaussianPolicy, MlpGaussianPolicy

# test general gaussian policy
# when observation is one dimension
actor = GeneralGaussianPolicy(4, 2)
obs = torch.randn(1, 4)
output = actor.explore(obs)

# when observation is a image
actor = GeneralGaussianPolicy((1, 256, 256), 2)
obs = torch.rand(1, 1, 256, 256)
output = actor.explore(obs)

# when observation is one image adding some one dimensional features
actor = GeneralGaussianPolicy(((1, 256, 256), 4), 2)
obs = torch.rand(1, 1, 256, 256)
additional_obs = torch.randn(1, 4)
output = actor.explore(obs, additional_obs)


# test the MLP gaussian policy
actor = MlpGaussianPolicy(4, 2)
obs = torch.randn(1, 4)
output = actor.explore(obs)

# test convolution gaussian policy
actor = CnnGaussianPolicy((1, 256, 256), 2, ["relu", "softsign"])
obs = torch.rand(1, 1, 256, 256)
output = actor.explore(obs)
