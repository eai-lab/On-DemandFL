# [AISTATS 2024] On-Demand Federated Learning
This repository provides an implementation of the **On-Demand Federated Learning for Arbitrary Target Class Distributions**.
This code is tested in [pytorch docker image](nvcr.io/nvidia/pytorch:21.12-py3) with RTXA6000 and GeForce3090.

This repository executes on-demand federated learning on FashionMNIST, CIFAR-10, CIFAR-100. The following steps show the procedure for the On-Dem FL, which consists of 1) execution of backbone FL algorithm, 2) on-demand model training. The subdirectory contains bash scrips for execution.

- acknowledgement
    - https://github.com/ijgit/Federated-Averaging-PyTorch
    - https://github.com/KarhouTam/FL-bench
    - https://github.com/TsingZ0/PFL-Non-IID

## 1. Execution of Backbone FL Algorithm
Code in the `./baseline` directory includes code for training class distribution predictors, including FedAvg experiments. Experiment results will be used in on-demand fl experiments, so please do not change the directory or file name carelessly.
Execution commands can be found in the bash file named after each dataset, i.e., `fashionmnist.sh`, `cifar10.sh`, and `cifar100.sh`.

## 2. On-Demand Model Training
Code in the `./ondemfl` directory includes code for on-demand model training. Execution commands can be found in the bash file named after each dataset, i.e., `fashionmnist.sh`, `cifar10.sh`, and `cifar100.sh`.
Additionally, code in the `./ondemfl-gt` directory includes code for on-demand model training using the ground truth of class distribution of clients.

## 3. Other Federated Learning Algorithms
Code in the `./baseline` directory includes code for other federated learning algorithms, i.e., clustered federated learning (CFL), multi-task federated learning (FMTL), and personalized federated learning (FedFOMO).
