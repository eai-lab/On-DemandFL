import gc
from multiprocessing import reduction
import os
import pickle
import logging
from threading import local
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torch.utils.data import DataLoader

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
    def __init__(self, client_id, local_data, device, log_path):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.data = local_data
        self.dist = self.data.dist.reshape(1, -1)
        self.num_classes = self.dist.shape[1]
        self.log_path = log_path
        self.device = device
        self.round = 0
        self.__task_model = None
        self.task_model_param_list = None

    @property
    def task_model(self):
        return self.__task_model

    @task_model.setter
    def task_model(self, task_model):
        self.__task_model = task_model.to(self.device)

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.data)

    def setup(self, task_model_config):
        """Set up common configuration of each client; called by center server."""
        self.task_model_config = task_model_config
        if self.task_model_config['optimizer'] == 'SGD':
            self.task_model_config['optim_config'] = {'lr': task_model_config['lr'], 'momentum': task_model_config['momentum']}
        elif self.task_model_config['optimizer'] == 'Adam':
            self.task_model_config['optim_config'] = {'lr': task_model_config['lr']}

        self.dataloader = DataLoader(self.data, batch_size=self.task_model_config['bs'], shuffle=True)

    def softmax(self, arr):
        return arr/torch.sum(arr)

    def task_model_update(self):
        self.task_model_param_list = None

        '''update target local model using local dataset'''
        self.task_model.train()
        # self.task_model.to(self.device)

        optimizer = torch.optim.__dict__[self.task_model_config['optimizer']](self.task_model.parameters(), **self.task_model_config['optim_config'])
        for e in range(self.task_model_config['ep']):
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
  
                optimizer.zero_grad()
                preds = self.task_model(data)
                loss = torch.nn.__dict__[self.task_model_config['criterion']]()(preds, labels)

                loss.backward()
                optimizer.step() 

                # save model parameters (weight)
                model_param = [*dict(self.task_model.named_modules()).values()][-1].weight.grad.reshape(1, -1).clone().cpu().detach()

                if self.task_model_param_list == None:
                        self.task_model_param_list = model_param
                else:
                    self.task_model_param_list = torch.cat((self.task_model_param_list, model_param), dim = 0)

        #     if self.device != 'cpu': torch.cuda.empty_cache()

        # self.task_model.to("cpu")

    def client_update(self, round):
        self.round = round
        self.old_task_model = copy.deepcopy(self.task_model)
        self.task_model_update()

    def target_model_evaluate(self):
        self.task_model.eval()
        self.task_model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                preds = self.task_model(data)

                test_loss +=torch.nn.__dict__[self.tm_criterion]()(preds, labels).item()
                
                predicted = preds.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                if self.device != 'cpu': torch.cuda.empty_cache()
        self.task_model.to("cpu")

        test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)

        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> Test accuracy: {100. * test_accuracy:.2f}%\n"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        return test_loss, test_accuracy

    def model_evaluate(self, dataloader):
        self.task_model.eval()

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                preds = self.task_model(data)

                test_loss += torch.nn.__dict__[self.task_model_config['criterion']]()(preds, labels).item()
                
                predicted = preds.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

        test_loss = test_loss / len(dataloader)
        test_accuracy = correct / len(dataloader.dataset)

        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> Test accuracy: {100. * test_accuracy:.2f}%\n"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        return test_loss, test_accuracy

    # def client_evaluate(self):
    #     """Evaluate local model using local dataset (same as training set for convenience)."""
    #     task_loss, task_acc = self.target_model_evaluate()

    #     return task_loss, task_acc

    def test(self, dataloader_list):
        acc_list = []
        
        for dataloader in dataloader_list:
            loss, acc = self.model_evaluate(dataloader) 
            acc_list.append(acc)
            
        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test accuracy: {acc_list}\n"
        print(message, flush=True); logging.info(message)
        
        return acc_list