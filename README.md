# About this project
Missing data is very common in many datasets but traditional machine learning methods often can not solve the missing data problem greatly. In this project, we investigate the application of Gaussian Process (GP) and sparse GP algorithms for handling missing data in large datasets, focusing on a heart time series dataset from MIT-BIH. We demonstrate that the choice of kernel, specifically the RBF kernel, is critical for optimal performance, and that GP regression excels at filling in missing values in small intervals with limited training size. However, GP performance declines rapidly for larger interpolation and extrapolation intervals. Sparse GP models offer significant training speed advantages and require careful hyperparameter tuning to balance speed and accuracy. Our findings highlight the potential of GP and sparse GP models for missing value imputation and emphasize the importance of kernel selection and hyperparameter tuning for each specific case. You can find the detailed explanation of this project in our [reports](./PTSA_REPORT.pdf).

# Why GpyTorch
In this project, we build most of our GP models by GpyTorch. Here is the reason that we use GpyTorch: First in contrast to many existing GP packages, we do not provide full GP models for the user. Rather, we provide the tools necessary to quickly construct one. This is because we believe, analogous to building a neural network in standard PyTorch, it is important to have the flexibility to include whatever components are necessary. As can be seen in more complicated examples, this allows the user great flexibility in designing custom models. Here is a link for GpyTorch Regression Tutorial: https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html and a link for the GpyTorch Kernel document: https://docs.gpytorch.ai/en/latest/kernels.html.

# Explanation on Each File in This Project
The 2 most experiments are in Experiments A.ipynb and Experiment B.ipynb. They are about filling missing data with small intervals between each of the training data points (randomly split the data into  train and test subsets, use training set to fill in the missing values)

## Experiment A.ipynb
In this set of experiments, we are going to discuss how below the 3 factors influence the performance and training speed of our Gaussian process models. They are 1. kernel choice 2. inducing model choice 3. whether to include LOVE. To inverstigate the training speed of different inducing model choice, we use 1000 data points in this set of experiment.

## Experiment B.ipynb
In this set of experiments, we are conducting experiments to explore the train size and test size going to influence the performance and training speed of our Gaussian process models. Fixing the kernel choice as RBF or RBF + Periodic, inducing model as full GP, we range the testing size from 0.1 to 0.9 (No LOVE). To speed up the training process, we only use 600 data points in this set of experiments.

## Experiment Large Interval.ipynb and Experiment Large Interval with LOVE.ipynb
They are about filling missing data with large intervals between each of the training data points (put training set on the head and tail, get posterior inference on the missing value in the middle). In the later file, we apply Lanczos Variance Estimates (LOVE) for fast inference

## Experiment with Deep GP.ipynb
Add deep feature extractor to enhance the performance of filling missing data with large intervals

## Experiment with Sklearn.ipynb
Implement the GP model by Sklearn package, instead of GpyTorch. (This approach is simpler)

## GpyTorch Customerised Kernel.py
Compared with Sklearn package, the kernel choice in the GpyTorch is limited. If one wants to design more expressive kernel, they can build it in this file and the import the module when building GPRegressionModel.

## heart_rate_data.csv
Contains the heart rate data we collected from MIT-BIH.

## Sparse Gaussian Process for Missing Heart Rate Time.pdf
Report of our findings

# environment.yml
package version list for this project. 
