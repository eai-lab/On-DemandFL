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
        self.__dist_predictor_list = None
        self.task_model_param_list = None

    @property
    def task_model(self):
        return self.__task_model

    @property
    def dist_predictor_list(self):
        return self.__dist_predictor_list

    @task_model.setter
    def task_model(self, task_model):
        self.__task_model = task_model

    @dist_predictor_list.setter
    def dist_predictor_list(self, dist_predictor_list):
        """Local model setter for passing globally aggregated model parameters."""
        self.__dist_predictor_list = dist_predictor_list

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.data)

    def setup(self, arguments):
        """Set up common configuration of each client; called by center server."""
        self.arguments = arguments
        
        self.tm_optimizer = arguments['tm_optimizer']
        self.tm_criterion = arguments['tm_criterion']
        self.tm_bs = arguments['tm_bs']
        self.tm_ep = arguments['tm_ep']
        
        self.dm_optimizer = arguments['dm_optimizer']
        self.dm_criterion = arguments['dm_criterion']
        self.dm_bs = arguments['dm_bs']
        self.dm_ep = arguments['dm_ep']
        
        if self.tm_optimizer == 'SGD':
            self.tm_optim_config = {'lr': arguments['tm_lr'], 'momentum': arguments['tm_momentum']}
        elif self.tm_optimizer == 'Adam':
            self.tm_optim_config = {'lr': arguments['tm_lr']}

        if self.dm_optimizer == 'SGD':
            self.dm_optim_config = {'lr': arguments['dm_lr'], 'momentum': arguments['dm_momentum']}
        elif self.dm_optimizer == 'Adam':
            self.dm_optim_config = {'lr': arguments['dm_lr']}

        self.dataloader = DataLoader(self.data, batch_size=self.arguments['tm_bs'], shuffle=True)

    def softmax(self, arr):
        return arr/torch.sum(arr)

    def target_model_update(self):
        self.task_model_param_list = None
        self.local_dist_list = None

        '''update target local model using local dataset'''
        self.task_model.train()
        # self.task_model.to(self.device)

        optimizer = torch.optim.__dict__[self.tm_optimizer](self.task_model.parameters(), **self.tm_optim_config)
        for e in range(self.tm_ep):
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
  
                optimizer.zero_grad()
                preds = self.task_model(data)
                loss = torch.nn.__dict__[self.tm_criterion]()(preds, labels)

                loss.backward()
                optimizer.step() 

                # get distribution of batchs and model parameter for distriubion model training
                local_dist = torch.Tensor([
                    len(torch.where(labels == l)[0])
                    for l in range(0, self.num_classes)
                ])
                local_dist = self.softmax(local_dist).reshape(1, self.num_classes)
                if self.local_dist_list == None:
                    self.local_dist_list = local_dist
                else:
                    self.local_dist_list = torch.cat((self.local_dist_list, local_dist), dim = 0)

                # save model parameters (weight)
                model_param = [*dict(self.task_model.named_modules()).values()][-1].weight.grad.reshape(1, -1).clone() # .cpu().detach()
                
                if self.task_model_param_list == None:
                        self.task_model_param_list = model_param
                else:
                    self.task_model_param_list = torch.cat((self.task_model_param_list, model_param), dim = 0)

    def model_param_set(self, round):
        self.task_model_param_list = self.task_model_param_list.reshape(self.task_model_param_list.shape[0], self.task_model_param_list.shape[1])
        
        self.local_dist_dataset = TensorDataset(self.task_model_param_list, self.local_dist_list)
        self.local_dist_loader = DataLoader(self.local_dist_dataset, batch_size=self.dm_bs, shuffle=True, drop_last=True)

    def dist_predictor_update(self):
        '''update distribution local model using parameter of local model'''

        for dist_predictor in self.dist_predictor_list:
            
            optimizer = torch.optim.__dict__[self.dm_optimizer](dist_predictor.parameters(), **self.dm_optim_config)
            dist_predictor.train()

            for e in range(self.dm_ep):
                for model_param, local_dist in self.local_dist_loader:
                    model_param = model_param.to(self.device)
                    local_dist = local_dist.to(self.device)

                    optimizer.zero_grad()
                    preds = dist_predictor(model_param)

                    if self.dm_criterion == 'KLDivLoss':
                        loss = torch.nn.__dict__[self.dm_criterion](reduction='batchmean')(preds, local_dist)
                    elif self.dm_criterion == 'MSELoss':
                        loss = torch.nn.__dict__[self.dm_criterion]()(preds*self.num_classes, local_dist*self.num_classes)
                    else:
                        loss = torch.nn.__dict__[self.dm_criterion]()(preds, local_dist)

                    loss.backward()
                    optimizer.step()
                # try:
                #     for model_param, local_dist in self.local_dist_loader:

                #         model_param = model_param.to(self.device)
                #         local_dist = local_dist.to(self.device)

                #         optimizer.zero_grad()
                #         preds = dist_predictor(model_param)

                #         if self.dm_criterion == 'KLDivLoss':
                #             loss = torch.nn.__dict__[self.dm_criterion](reduction='batchmean')(preds, local_dist)
                #         elif self.dm_criterion == 'MSELoss':
                #             loss = torch.nn.__dict__[self.dm_criterion]()(preds*100, local_dist*100)
                #         else:
                #             loss = torch.nn.__dict__[self.dm_criterion]()(preds, local_dist)

                #         loss.backward()
                #         optimizer.step()
                # except:
                #     print('error occur')


    def client_update(self, round):
        self.round = round
        self.target_model_update()
        self.model_param_set(self.round)
        self.dist_predictor_update()

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

        test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)

        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> Test accuracy: {100. * test_accuracy:.2f}%\n"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        return test_loss, test_accuracy
