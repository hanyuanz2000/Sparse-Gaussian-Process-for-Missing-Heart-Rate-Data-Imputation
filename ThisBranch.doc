## About SKI
http://proceedings.mlr.press/v37/wilson15.pdf

We introduce a new structured kernel interpolation (SKI) framework, which generalises
and unifies inducing point methods for scalable Gaussian processes (GPs). SKI methods produce kernel approximations for fast computations through kernel interpolation. The SKI framework clarifies how the quality of an inducing point approach depends on the number of inducing (aka interpolation) points, interpolation strategy, and GP covariance kernel. Using SKI, with local cubic kernel interpolation, we introduce KISSGP, which is 1) more scalable than inducing point alternatives, 2) naturally enables Kronecker and Toeplitz algebra for substantial additional gains in scalability, without requiring any grid data,
and 3) can be used for fast and expressive kernel learning. KISS-GP costs O(n) time and storage for GP inference. We evaluate KISS-GP for kernel matrix approximation, kernel learning, and natural sound modelling.

## why gpytorch
First in contrast to many existing GP packages, we do not provide full GP models for the user. Rather, we provide the tools necessary to quickly construct one. This is because we believe, analogous to building a neural network in standard PyTorch, it is important to have the flexibility to include whatever components are necessary. As can be seen in more complicated examples, this allows the user great flexibility in designing custom models.

https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html

### gpytorch kernel
https://docs.gpytorch.ai/en/latest/kernels.html

Kernels in GPyTorch are implemented as a gpytorch.Module that, when called on two torch.Tensor objects  x_1 and x_2, returns either a torch.Tensor or a LinearOperator that represents the covariance matrix between x_1 and x_2.