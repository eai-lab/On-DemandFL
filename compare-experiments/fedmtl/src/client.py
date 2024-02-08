import gc
from multiprocessing import reduction
import os
import pickle
import logging
from threading import local

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torch.utils.data import DataLoader
import copy
import math

logger = logging.getLogger(__name__)

def flatten(model):
    state_dict = model.state_dict()
    keys = state_dict.keys()
    W = [state_dict[key].flatten() for key in keys]
    return torch.cat(W)

class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """
    def __init__(self, client_id, local_data, device, log_path):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.data = local_data
        self.dist = self.data.dist.reshape(1, -1)
        self.num_classes = self.dist.shape[1]
        self.log_path = log_path
        self.device = device
        self.round = 0
        self.__model = None
        self.model_param_list = None
                
        self.omega = None
        self.W_glob = None
        self.idx = 0
        self.lamba = 1e-4


    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model.to(self.device)
        
    def set_model(self, model):
        self.model.load_state_dict(model.state_dict())
        
    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.data)

    def setup(self, model_config, model):
        """Set up common configuration of each client; called by center server."""
        self.model_config = model_config
        if self.model_config['optimizer'] == 'SGD':
            self.model_config['optim_config'] = {'lr': model_config['lr'], 'momentum': model_config['momentum']}
        elif self.model_config['optimizer'] == 'Adam':
            self.model_config['optim_config'] = {'lr': model_config['lr']}
        self.model = copy.deepcopy(model)
        
        self.dataloader = DataLoader(self.data, batch_size=self.model_config['bs'], shuffle=True)

    def softmax(self, arr):
        return arr/torch.sum(arr)

    def model_update(self):
        self.model_param_list = None

        '''update target local model using local dataset'''
        self.model.train()
        # self.model.to(self.device)

        optimizer = torch.optim.__dict__[self.model_config['optimizer']](self.model.parameters(), **self.model_config['optim_config'])
        for e in range(self.model_config['ep']):
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
  
                optimizer.zero_grad()
                preds = self.model(data)
                loss = torch.nn.__dict__[self.model_config['criterion']]()(preds, labels)

                self.W_glob[:, self.idx] = flatten(self.model)
                loss_regularizer = 0
                loss_regularizer += self.W_glob.norm() ** 2

                loss_regularizer += torch.sum(torch.sum((self.W_glob*self.omega), 1)**2)
                f = (int)(math.log10(self.W_glob.shape[0])+1) + 1
                loss_regularizer *= 10 ** (-f)
                loss += loss_regularizer
                
                loss.backward()
                optimizer.step() 

    def set_parameters(self, W_glob, omega, idx):
        self.omega = torch.sqrt(omega[0][0])
        self.W_glob = copy.deepcopy(W_glob)
        self.idx = idx

    def client_update(self, round):
        self.round = round
        self.model_update()
        
    def finetune(self):
        self.model.train()
        optimizer = torch.optim.__dict__[self.model_config['optimizer']](self.model.parameters(), **self.model_config['optim_config'])
        
        for _ in range(self.model_config['finetune_ep']):
            for data, labels in self.dataloader:
                if len(data) <= 1:
                    continue
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                logit = self.model(data)
                loss = torch.nn.__dict__[self.model_config['criterion']]()(logit, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def test(self, dataloader_list):
        # self.set_model(model_param)
        # self.model.load_state_dict(model_param)
        # before_loss, before_acc = self.model_evaluate(dataloader)
        
        # self.finetune()
        acc_list = []
        
        for dataloader in dataloader_list:
            loss, acc = self.model_evaluate(dataloader) 
            acc_list.append(acc)
            
        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test accuracy: {acc_list}\n"
        print(message, flush=True); logging.info(message)
        
        return acc_list


    def model_evaluate(self, dataloader):
        self.model.eval()

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                preds = self.model(data)

                test_loss += torch.nn.__dict__[self.model_config['criterion']]()(preds, labels).item()
                
                predicted = preds.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

        test_loss = test_loss / len(dataloader)
        test_accuracy = correct / len(dataloader.dataset)

        return test_loss, test_accuracy
    
    