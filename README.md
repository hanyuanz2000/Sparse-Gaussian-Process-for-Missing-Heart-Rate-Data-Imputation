# About this project
Missing data is very common in many datasets but traditional machine learning methods often can not solve the missing data problem greatly. In this project, we try to apply Gaussian Process(GP) to solve this problem. Gaussian Process(GP) is a nonparametric method with uncertainty estimation. However, For large datasets, full GP is slow. Hence, we
combine the GPs with sparse approximation to get some sparse GP algorithms that can make the computation tractable. In this project, We will carry out empirical studys to investigate the performance of full GP as well as different sparse GPs to a heart time series dataset from MIT-BIH, with different kernel choice, gp models, and training size. We get lots of interesting findings, including SKI and SGPR doesn't really speed up the training process compared with full GP in small data sets. Also, GP is powerful for small interval interpolation, even with a very small training size. However, when it comes to large interval interpolation or extrapolation, even deep GP can't provides a satisfactory inference result.

# Why GpyTorch
In this project, we build most of our GP models by GpyTorch. Here is the reason that we use GpyTorch: First in contrast to many existing GP packages, we do not provide full GP models for the user. Rather, we provide the tools necessary to quickly construct one. This is because we believe, analogous to building a neural network in standard PyTorch, it is important to have the flexibility to include whatever components are necessary. As can be seen in more complicated examples, this allows the user great flexibility in designing custom models. Here is a link for GpyTorch Regression Tutorial: https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html and a link for the GpyTorch Kernel document: https://docs.gpytorch.ai/en/latest/kernels.html.

# Explanation on Each File in This Project



