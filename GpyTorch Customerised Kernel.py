#This file is for design customer kernel. We can import kernels from this file
#when we try to build gpytorch model and need some customer kernel
import math
import torch
import gpytorch
from matplotlib import pyplot as plt

class FirstSincKernel(gpytorch.kernels.Kernel):
    # the sinc kernel is stationary
    is_stationary = True

    # this is the kernel function
    def forward(self, x1, x2, **params):
        # calculate the distance between inputs
        diff = self.covar_dist(x1, x2, **params)[0][0]
        print('before where:', diff)
        # prevent divide by 0 errors
        diff = diff.where(diff < 1e-20, torch.as_tensor(3))
        print('after where:', diff)
        # return sinc(diff) = sin(diff) / diff
        return torch.sin(diff).div(diff)