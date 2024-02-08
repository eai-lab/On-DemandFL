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
import numpy as np


logger = logging.getLogger(__name__)


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
    def __init__(self, client_id, local_data, val_data, device, num_clients, log_path):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.train_data = local_data
        self.val_data = val_data
        self.dist = self.train_data.dist.reshape(1, -1)
        self.num_classes = self.dist.shape[1]
        self.log_path = log_path
        self.device = device
        self.round = 0
        self.__model = None
        self.model_param_list = None
        
        # for fedfomo
        self.received_ids = []
        self.received_models = []
        self.num_clients = num_clients
        self.weight_vector = torch.zeros(self.num_clients, device=self.device)
        
        # self.train_samples = self.train_samples * (1-self.val_ratio)


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
        return len(self.train_data)

    def setup(self, model_config, model):
        """Set up common configuration of each client; called by center server."""
        self.model_config = model_config
        if self.model_config['optimizer'] == 'SGD':
            self.model_config['optim_config'] = {'lr': model_config['lr'], 'momentum': model_config['momentum']}
        elif self.model_config['optimizer'] == 'Adam':
            self.model_config['optim_config'] = {'lr': model_config['lr']}
        
        self.model = copy.deepcopy(model)
        self.old_model = copy.deepcopy(self.model)
        
        self.trainloader = DataLoader(self.train_data, batch_size=self.model_config['bs'], shuffle=True)
        self.valloader = DataLoader(self.val_data, batch_size=self.model_config['bs'], shuffle=True)


    def softmax(self, arr):
        return arr/torch.sum(arr)

    def model_update(self):

        '''update target local model using local dataset'''
        self.model.train()
        # self.model.to(self.device)

        optimizer = torch.optim.__dict__[self.model_config['optimizer']](self.model.parameters(), **self.model_config['optim_config'])
        for e in range(self.model_config['ep']):
            for data, labels in self.trainloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
  
                optimizer.zero_grad()
                preds = self.model(data)
                loss = torch.nn.__dict__[self.model_config['criterion']]()(preds, labels)

                loss.backward()
                optimizer.step() 

        #     if self.device != 'cpu': torch.cuda.empty_cache()

        # self.model.to("cpu")

    def client_update(self, round):
        self.round = round
        # self.model_update()
        
        self.aggregate_parameters(self.valloader)
        self.old_model = copy.deepcopy(self.model)
        
        self.model_update()
        
        
    def finetune(self):
        self.model.train()
        optimizer = torch.optim.__dict__[self.model_config['optimizer']](self.model.parameters(), **self.model_config['optim_config'])
        
        for _ in range(self.model_config['finetune_ep']):
            for data, labels in self.trainloader:
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

        # message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
        #     \n\t=> Test loss: {test_loss:.4f}\
        #     \n\t=> Test accuracy: {100. * test_accuracy:.2f}%\n"
        # print(message, flush=True); logging.info(message)
        # del message; gc.collect()

        return test_loss, test_accuracy

    
    # def dist_model_evaluate(self):
        # pass

    # def client_evaluate(self):
    #     """Evaluate local model using local dataset (same as training set for convenience)."""
    #     task_loss, task_acc = self.target_model_evaluate()

    #     return task_loss, task_acc


    # for FedFomo
    
    def recalculate_loss(self, new_model, val_loader):
        L = 0
        for x, y in val_loader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            output = new_model(x)
            # loss = self.loss(output, y)
            
            loss = torch.nn.__dict__[self.model_config['criterion']]()(output, y.long())
            
            
            L += loss.item()
        return L / len(val_loader)
    
    def weight_vector_update(self, weight_list):
        # self.weight_vector = torch.zeros(self.num_clients, device=self.device)
        # for w, id in zip(weight_list, self.received_ids):
        #     self.weight_vector[id] += w.clone()
    
        self.weight_vector = np.zeros(self.num_clients)
        for w, id in zip(weight_list, self.received_ids):
            self.weight_vector[id] += w.item()
        self.weight_vector = torch.tensor(self.weight_vector).to(self.device)
        
    
    def weight_cal(self, val_loader):
        weight_list = []
        L = self.recalculate_loss(self.old_model, val_loader)
        for received_model in self.received_models:
            params_dif = []
            for param_n, param_i in zip(received_model.parameters(), self.old_model.parameters()):
                params_dif.append((param_n - param_i).view(-1))
            params_dif = torch.cat(params_dif)

            weight_list.append((L - self.recalculate_loss(received_model, val_loader)) / (torch.norm(params_dif) + 1e-5))
        self.weight_vector_update(weight_list)

        return torch.tensor(weight_list)
    
    def weight_scale(self, weights):
        weights = torch.maximum(weights, torch.tensor(0))
        w_sum = torch.sum(weights)
        if w_sum > 0:
            weights = [w/w_sum for w in weights]
            return torch.tensor(weights)
        else:
            return torch.tensor([])

    def add_parameters(self, w, received_model):
        for param, received_param in zip(self.model.parameters(), received_model.parameters()):
            param.data += received_param.data.clone() * w
            
    def aggregate_parameters(self, val_loader):
        weights = self.weight_scale(self.weight_cal(val_loader))
        
        if len(weights) > 0:
            for param in self.model.parameters():
                param.data.zero_()
                
            for w, received_model in zip(weights, self.received_models):
                self.add_parameters(w, received_model)
    