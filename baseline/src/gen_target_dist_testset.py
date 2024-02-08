from random import random

import numpy as np
from tqdm import tqdm

import cv2 as cv
import imageio
import random
import math
import pandas as pd

import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, TensorDataset, DataLoader
import os



class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, data, targets, class_to_idx, transform=None):

        # assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.data = data
        self.targets = targets
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __getitem__(self, index):
        if type(self.data[index]) == str:
            data = torch.Tensor(imageio.imread(self.data[index], as_gray=False, pilmode="RGB"))
        else:
            data = self.data[index]
        targets = self.targets[index]

        if self.transform:
            data = self.transform(data.numpy().astype(np.uint8))
        return data, targets

    def __data__(self):
        return self.data
    
    def __targets__(self):
        return self.targets
   
    def __class_to_idx__(self):
        return self.class_to_idx

    def __len__(self):
        return len(self.data) if type(self.data) == list else self.data.size(0)


def gen_target_distribution_data(targets, class_to_idx, dist, target_dist_op):
    num_of_data = len(targets)//len(class_to_idx.values())
    sorted_indecies = torch.argsort(torch.Tensor(targets))
    sorted_labels = torch.Tensor(targets)[sorted_indecies]
    
    num_of_class_item = {}
    for i in class_to_idx.values():
        num_of_class_item[i] = len(sorted_labels[sorted_labels==i])
        
    new_indecies = None
    init_idx = 0

    if target_dist_op == 4:
        min_value = min(num_of_class_item)
        for i in class_to_idx.values():
            if new_indecies == None:
                new_indecies = sorted_indecies[init_idx: min_value]
            else:
                new_indecies = torch.cat([new_indecies, sorted_indecies[init_idx: min_value]], dim=0)
            
            init_idx = init_idx + num_of_class_item[i]
    else:
        new_indecies = None
        init_idx = 0
        for i in class_to_idx.values():
            if new_indecies == None:
                new_indecies = sorted_indecies[init_idx: init_idx+math.ceil(dist[i]*num_of_data)]
            else:
                new_indecies = torch.cat([new_indecies, sorted_indecies[init_idx: init_idx+math.ceil(dist[i]*num_of_data)]], dim=0)
            
            init_idx = init_idx + num_of_class_item[i]

        return new_indecies


def prepare_dataset(seed, dataset_name, target_dist, target_dist_op):
    dataset_name = dataset_name.upper()

    if hasattr(torchvision.datasets, dataset_name):
        if dataset_name == "MNIST" or dataset_name == "EMNIST":
            transform = transforms.Compose(
                [
                    transforms.ToTensor()
                ]
            )
        
        elif dataset_name == "CIFAR10" or dataset_name == "CIFAR100":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )

        if dataset_name in ["EMNIST"]:
            train_dataset = torchvision.datasets.__dict__[dataset_name](root='./data', train=True, split="byclass",
                                                download=True, transform=transform)
            test_dataset = torchvision.datasets.__dict__[dataset_name](root='./data', train=False, split="byclass",
                                                download=True, transform=transform)
        else:
            train_dataset = torchvision.datasets.__dict__[dataset_name](root='./data', train=True,
                                                download=True, transform=transform)
            test_dataset = torchvision.datasets.__dict__[dataset_name](root='./data', train=False,
                                                download=True, transform=transform)

        if "ndarray" not in str(type(train_dataset.data)):
            train_dataset.data = np.asarray(train_dataset.data)
            test_dataset.data = np.asarray(test_dataset.data)
        if "list" not in str(type(train_dataset.targets)):
            train_dataset.targets = train_dataset.targets.tolist()
            test_dataset.targets = test_dataset.targets.tolist()
        

        # split test dataset into validation set and test set
        test_dataset_indecies = gen_target_distribution_data(test_dataset.targets, test_dataset.class_to_idx, target_dist, target_dist_op)
        test_dataset = CustomTensorDataset(
            data = torch.Tensor(test_dataset.data)[test_dataset_indecies],
            targets = torch.Tensor(test_dataset.targets)[test_dataset_indecies],
            class_to_idx = test_dataset.class_to_idx,
            transform=transform
        )

    return test_dataset