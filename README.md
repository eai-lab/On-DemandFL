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

## Citation (BibTeX)

**On-Demand Federated Learning for Arbitrary Target Class Distributions**

```
@InProceedings{pmlr-v238-jeong24a,
  title = 	 { On-Demand Federated Learning for Arbitrary Target Class Distributions },
  author =       {Jeong, Isu and Lee, Seulki},
  booktitle = 	 {Proceedings of The 27th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {3421--3429},
  year = 	 {2024},
  editor = 	 {Dasgupta, Sanjoy and Mandt, Stephan and Li, Yingzhen},
  volume = 	 {238},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {02--04 May},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v238/jeong24a/jeong24a.pdf},
  url = 	 {https://proceedings.mlr.press/v238/jeong24a.html},
  abstract = 	 { We introduce On-Demand Federated Learning (On-Demand FL), which enables on-demand federated learning of a deep model for an arbitrary target data distribution of interest by making the best use of the heterogeneity (non-IID-ness) of local client data, unlike existing approaches trying to circumvent the non-IID nature of federated learning. On-Demand FL composes a dataset of the target distribution, which we call the composite dataset, from a selected subset of local clients whose aggregate distribution is expected to emulate the target distribution as a whole. As the composite dataset consists of a precise yet diverse subset of clients reflecting the target distribution, the on-demand model trained with exactly enough selected clients becomes able to improve the model performance on the target distribution compared when trained with off-target and/or unknown distributions while reducing the number of participating clients and federating rounds. We model the target data distribution in terms of class and estimate the class distribution of each local client from the weight gradient of its local model. Our experiment results show that On-Demand FL achieves up to 5% higher classification accuracy on various target distributions just involving 9${\times}$ fewer clients with FashionMNIST, CIFAR-10, and CIFAR-100. }
}
```