import math
import torch
import gpytorch
from matplotlib import pyplot as plt

class WhiteKernel(gpytorch.kernels.Kernel):
    # the WhiteKernel kernel is stationary
    is_stationary = True
    
    def __init__(self, noise_level) -> None:
        super().__init__()
        self.noise_level = noise_level

    # this is the kernel function
    def forward(self, x1, x2, **params):
        # calculate the distance between inputs
        diff = self.covar_dist(x1, x2, **params)
        b = diff.where(diff == 0, torch.as_tensor(3))
        # diff = diff.where(diff < 1e-20, torch.as_tensor(1e-20))
        return b

class FirstSincKernel(gpytorch.kernels.Kernel):
    # the sinc kernel is stationary
    is_stationary = True

    # this is the kernel function
    def forward(self, x1, x2, **params):
        # calculate the distance between inputs
        diff = self.covar_dist(x1, x2, **params)
        print('diff1', diff)
        # prevent divide by 0 errors
        diff.where(diff == 0, 2*diff)
        print('diff2', diff)
        # return sinc(diff) = sin(diff) / diff
        return torch.sin(diff).div(diff)