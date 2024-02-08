from random import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

   
# split_noniid(train_dataset., train_dataset.targets)
def split_noniid(train_idcs, train_labels, alpha, n_clients, seed):
    '''
    Splits a list of data indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter
    alpha
    '''
    # print(train_idcs)
    train_idcs = np.array(train_idcs)
    train_labels = np.array(train_labels)

    n_classes = max(train_labels)+1
    label_distribution = np.random.default_rng(seed=seed).dirichlet(alpha=np.repeat(alpha,n_clients), size=n_classes)
    # label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels[train_idcs]==y).flatten() for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]
  

    dist = np.zeros((n_clients, n_classes))
    for i in range(0, n_clients):
        for j in client_idcs[i]:
            dist[i, train_labels[j]] += 1
    return client_idcs, dist


def partition_dataset(targets, class_to_idx, num_client, alpha, seed):
    num_class = len(class_to_idx.values())

    num_of_data = len(targets)
    sorted_indecies = torch.argsort(torch.Tensor(targets))
    sorted_labels = torch.Tensor(targets)[sorted_indecies]

    # counts the number of labels corresponding to the index.
    num_of_class_item = {}
    for i in class_to_idx.values():
        num_of_class_item[i] = len(sorted_labels[sorted_labels==i])

    # store the index corresponding to each label
    sorted_class_indecies = {}

    init_idx = 0
    for idx in num_of_class_item.keys():
        sorted_class_indecies[idx] = sorted_indecies[init_idx:init_idx+num_of_class_item[idx]]
        init_idx += num_of_class_item[idx]
        
        
    ''' split dataset into clients via label'''
    # init client_data_idx
    client_data_indecies  = {}
    count_client_data = np.zeros((num_client, num_class))
    for i in range(0, num_client):
        client_data_indecies[i] = None

    dist = np.random.default_rng(seed=seed).dirichlet(alpha=np.repeat(alpha,num_client), size=num_class)

    for class_i, class_dist in zip(range(0, num_class), dist):
        indecies = sorted_class_indecies[class_i]
        num_each_class = len(indecies)
        init_idx = 0


        for client_i in range(0, num_client):
            client_data_idx = init_idx + round(class_dist[client_i]*len(indecies))
            
            if init_idx >= num_each_class:
                break

            if client_data_idx > num_each_class:
                client_data_idx = num_each_class
                
            # if client_i == num_client-1 and client_data_idx > num_each_class:
            #    client_data_idx = num_each_class

            count_client_data[client_i][class_i] = client_data_idx-init_idx

            if client_data_indecies[client_i] == None:
                client_data_indecies[client_i] = indecies[init_idx: client_data_idx]
            else:
                client_data_indecies[client_i] = torch.cat([client_data_indecies[client_i], indecies[init_idx: client_data_idx]], dim=0)
            init_idx = client_data_idx

    return client_data_indecies, count_client_data, dist.T


class ClientDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, data, targets, dist, class_to_idx, transform=None):

        # assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.data = data
        self.targets = targets
        self.dist = dist
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __getitem__(self, index):
        if type(self.data[index]) == str:
            data = torch.Tensor(imageio.imread(self.data[index], as_gray=False, pilmode="RGB"))
        else:
            data = self.data[index]
        targets = self.targets[index]
        if self.transform:
            data = self.transform(data)
        return data, targets

    def __dist__(self):
        return self.dist
    
    def __data__(self):
        return self.data
    
    def __targets__(self):
        return self.targets

    def __class_to_idx__(self):
        return self.class_to_idx

    def __len__(self):
        return self.data.shape[0]
        
        #len(self.data) if type(self.data) == list else self.data.size(0)

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
            data = self.transform(data)
        return data, targets

    def __data__(self):
        return self.data
    
    def __targets__(self):
        return self.targets
   
    def __class_to_idx__(self):
        return self.class_to_idx

    def __len__(self):
        return self.data.shape[0] #len(self.data)# if type(self.data) == list else self.data.size(0)



def partition_with_dirichlet_distribution(dataset_name, data, targets, class_to_idx, num_client, alpha, transform, seed):

    idcs = [i for i in range(0, len(data.data))]
    client_data_indecies, client_dist = split_noniid(idcs, targets, alpha, num_client, seed)

    def softmax(arr):
        return torch.Tensor(arr/np.sum(arr))

    splited_client_dataset = [
        ClientDataset(
            data = data[client_data_indecies[idx]], # client_datas[idx] if type(data[0]) == str else torch.Tensor(client_datas[idx]),
            targets = torch.Tensor(targets)[client_data_indecies[idx]],
            dist = softmax(client_dist[idx]),
            class_to_idx = class_to_idx,
            transform=transform
        )
        for idx in range(0, num_client)
    ]
    
    return splited_client_dataset


def split_dataset_with_test_and_val(dataset):
    sorted_indecies = torch.argsort(torch.Tensor(dataset.targets))
    sorted_labels = torch.Tensor(dataset.targets)[sorted_indecies]
    
    num_of_class_item = {}
    for i in dataset.class_to_idx.values():
        num_of_class_item[i] = len(sorted_labels[sorted_labels==i])
    
    val_dataset_indecies, test_dataset_indecies = None, None
    init_idx = 0
    for i in dataset.class_to_idx.values():
        val_idx = num_of_class_item[i]//2
        test_idx = num_of_class_item[i] 
        
        if val_dataset_indecies==None and test_dataset_indecies==None:
            val_dataset_indecies = sorted_indecies[init_idx:init_idx+val_idx]
            test_dataset_indecies = sorted_indecies[init_idx+val_idx:init_idx+test_idx]
        else:
            val_dataset_indecies = torch.cat([val_dataset_indecies, sorted_indecies[init_idx:init_idx+val_idx]], dim=0)
            test_dataset_indecies = torch.cat([test_dataset_indecies, sorted_indecies[init_idx+val_idx:init_idx+test_idx]], dim=0)
        
        init_idx += test_idx
        
    
    return val_dataset_indecies, test_dataset_indecies
    

def gen_target_distribution_data(targets, class_to_idx, dist):
    num_of_data = len(targets)//len(class_to_idx.values())
    sorted_indecies = torch.argsort(torch.Tensor(targets))
    sorted_labels = torch.Tensor(targets)[sorted_indecies]
    
    num_of_class_item = {}
    for i in class_to_idx.values():
        num_of_class_item[i] = len(sorted_labels[sorted_labels==i])
        
    new_indecies = None
    init_idx = 0
    for i in class_to_idx.values():
        if new_indecies == None:
            new_indecies = sorted_indecies[init_idx: init_idx+round(dist[i]*num_of_data)]
        else:
            new_indecies = torch.cat([new_indecies, sorted_indecies[init_idx: init_idx+round(dist[i]*num_of_data)]], dim=0)
        
        init_idx = init_idx + num_of_class_item[i]

    return new_indecies


def prepare_dataset(seed, dataset_name, target_dist_list, num_client, alpha):
    dataset_name = dataset_name.upper()
    
    if dataset_name == "FASHIONMNIST":
        dataset_name = "FashionMNIST"

    if hasattr(torchvision.datasets, dataset_name):
        # if dataset_name == "MNIST" or dataset_name == "EMNIST" or dataset_name == "SVHN":
        #     transform = transforms.Compose(
        #         [
        #             transforms.ToTensor()
        #         ]
        #     )
        
        # elif dataset_name == "CIFAR10" or dataset_name == "CIFAR100":
        #     transform = transforms.Compose(
        #         [
        #             transforms.ToTensor(),
        #             # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #         ]
        #     )
        transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )

        if dataset_name in ["EMNIST"]:
            train_dataset = torchvision.datasets.__dict__[dataset_name](root='/workspace/shared/data', train=True, split="byclass",
                                                download=True)#, transform=transform)
            test_dataset = torchvision.datasets.__dict__[dataset_name](root='/workspace/shared/data', train=False, split="byclass",
                                                download=True)#, transform=transform)
        elif dataset_name in ["SVHN"]:
            train_dataset = torchvision.datasets.__dict__[dataset_name](root='/workspace/shared/data', split="train",
                                                download=True)#, transform=transform)
            test_dataset = torchvision.datasets.__dict__[dataset_name](root='/workspace/shared/data',  split="test",
                                                download=True)#, transform=transform)
            train_dataset.data = np.transpose(train_dataset.data, (0, 2, 3, 1))
            test_dataset.data = np.transpose(test_dataset.data, (0, 2, 3, 1))
            train_dataset.targets = train_dataset.labels
            test_dataset.targets = test_dataset.labels
            train_dataset.class_to_idx = {}; test_dataset.class_to_idx = {}
            for i in range(0, 10):
                train_dataset.class_to_idx[f"{i}"] = i
                test_dataset.class_to_idx[f"{i}"] = i
        else:
            train_dataset = torchvision.datasets.__dict__[dataset_name](root='/workspace/shared/data', train=True,
                                                download=True)#, transform=transform)
            test_dataset = torchvision.datasets.__dict__[dataset_name](root='/workspace/shared/data', train=False,
                                                download=True)#, transform=transform)

        if "ndarray" not in str(type(train_dataset.data)):
            train_dataset.data = np.asarray(train_dataset.data)
            test_dataset.data = np.asarray(test_dataset.data)
        if "list" not in str(type(train_dataset.targets)):
            train_dataset.targets = train_dataset.targets.tolist()
            test_dataset.targets = test_dataset.targets.tolist()

        # split test dataset into validation set and test set
        test_dataset_list = []
        # testset = CustomTensorDataset(test_dataset.data, torch.Tensor(test_dataset.targets), test_dataset.class_to_idx, transform=transform)
        # test_dataset_list.append(testset)
        
        idcs = np.array([i for i in range(0, len(test_dataset.data))])
        labels = np.array(test_dataset.targets)
        n_classes = max(labels)+1
        class_idcs = [np.argwhere(labels[idcs]==y).flatten() for y in range(n_classes)]
        
        class_idcs_list = []
        for i in range(n_classes-1, 0, -1):
            _class_idcs = np.concatenate([class_idcs[i] for i in range(0, i+1)])
            class_idcs_list.append(_class_idcs)
        class_idcs_list.append(class_idcs[0])
        class_idcs_list = class_idcs_list[::-1]

        for target_idcs in class_idcs_list:
            testset = CustomTensorDataset(
                data = test_dataset.data[target_idcs],
                targets = torch.Tensor(test_dataset.targets)[target_idcs],
                class_to_idx = test_dataset.class_to_idx,
                transform=transform
            )
            test_dataset_list.append(testset)

        for target_dist in target_dist_list:
            testset_indecies = gen_target_distribution_data(test_dataset.targets, test_dataset.class_to_idx, target_dist)
            testset = CustomTensorDataset(
                data = test_dataset.data[testset_indecies],
                targets = torch.Tensor(test_dataset.targets)[testset_indecies],
                class_to_idx = test_dataset.class_to_idx,
                transform=transform
            )
            test_dataset_list.append(testset)

        partitioned_train_set = partition_with_dirichlet_distribution(dataset_name, train_dataset.data, train_dataset.targets, train_dataset.class_to_idx, num_client, alpha, transform, seed)
        
    return partitioned_train_set, test_dataset_list

