from cProfile import label
import copy
import gc
import logging
from os import system
from pyexpat import model

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torchvision

from multiprocessing import pool, cpu_count
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import OrderedDict
import pickle

from .models import *
from .utils import *
from .client import Client
from scipy.optimize import nnls
logger = logging.getLogger(__name__)


class Server(object):
    """Class for implementing center server orchestrating the whole process of federated learning
    
    At first, center server distribute model skeleton to all participating clients with configurations.
    While proceeding federated learning rounds, the center server samples some fraction of clients,
    receives locally updated parameters, averages them as a global parameter (model), and apply them to global model.
    In the next round, newly selected clients will recevie the updated global model as its local model.  

    """

    def __init__(self, writer, train_dataset, test_dataset_list, fed_config={}, data_config={}, model_config={}, system_config={}):
        self.clients = None
        self._round = 0
        self.writer = writer

        # dataset 
        self.train_dataset = train_dataset
        self.test_dataset_list = test_dataset_list
        self.data_config = data_config
        self.num_classes = int(data_config['num_classes'])
        
        # system setting
        self.device = system_config['device']
        self.mp_flag = system_config['is_mp']
        self.log_dir = system_config['log_dir']

        # federated setting
        self.num_clients = fed_config['num_clients']
        self.num_rounds = fed_config['num_rounds']
        self.fraction = fed_config['fraction']

        # gen model
        self.model_config = model_config
        self.model = eval(model_config['name'])().to(self.device)
        message = f"{self.model}\n"
        print(message); logging.info(message)
        del message; gc.collect()
        
        # fedMTL
        self.join_clients = int(self.num_clients*self.fraction)
        
        self.dim = len(self.flatten(self.model))
        self.W_glob = torch.zeros((self.dim, self.join_clients), device=self.device)
        I = torch.ones((self.join_clients, self.join_clients))
        i = torch.ones((self.join_clients, 1))
        omega = (I - 1 / self.join_clients * i.mm(i.T)) ** 2
        self.omega = omega.to(self.device)

        
    def setup(self, **init_kwargs):
        """Set up all configuration for federated learning."""
        # valid only before the very first round
        assert self._round == 0

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.model.parameters()))})!"
        print(message); logging.info(message)
        del message; gc.collect()

        # assign dataset to each client
        self.clients = self.create_clients(self.train_dataset)
        
        # save client's distribution for debuging
        self.client_dist = []
        for c in self.clients:
            self.client_dist.append(c.dist)
        with open(f"./{self.log_dir}/{self.data_config['name']}-a{self.data_config['alpha']}-client_distribution.pkl", 'wb') as f:
            pickle.dump(self.client_dist, f)

        # prepare hold-out dataset for evaluation
        self.testloader_list = [
            DataLoader(testset, batch_size=256, shuffle=False)
            for testset in self.test_dataset_list
        ]

        # configure detailed settings for client upate and 
        self.setup_clients(
            model_config = self.model_config,
            model = self.model
        )
        
        # send the model skeleton to all clients
        self.transmit_model()
        
    def create_clients(self, local_datasets):
        """Initialize each Client instance."""
        clients = []
        for k, dataset in tqdm(enumerate(local_datasets), leave=False):
            client = Client(client_id=k, local_data=dataset, device=self.device, log_path=f"{self.log_dir}")
            clients.append(client)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully created all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()
        return clients

    def setup_clients(self, **client_config):
        """Set up each client."""
        for k, client in tqdm(enumerate(self.clients), leave=False):
            client.setup(**client_config)
        
        message = f"[Round: {str(self._round).zfill(4)}] ...successfully finished setup of all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def transmit_model(self, sampled_client_indices=None):
        """Send the updated global model to selected/all clients."""
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            assert (self._round == 0) or (self._round == self.num_rounds)

            for client in tqdm(self.clients, leave=False):
                client.set_model(self.model)

            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
            print(message); logging.info(message)
            del message; gc.collect()
        else:
            # send the global model to selected clients
            assert self._round != 0

            for idx in tqdm(sampled_client_indices, leave=False):
                self.clients[idx].set_model(self.model)
            
            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!"
            print(message); logging.info(message)
            del message; gc.collect()

    def sample_clients(self):
        """Select some fraction of all clients."""
        # sample clients randommly
        message = f"[Round: {str(self._round).zfill(4)}] Select clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients, replace=False).tolist())

        return sampled_client_indices
    
    def update_selected_clients(self, sampled_client_indices):
        """Call "client_update" function of each selected client."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        selected_total_size = 0
        for idx in tqdm(sampled_client_indices, leave=False):
            self.clients[idx].client_update(self._round)
            selected_total_size += len(self.clients[idx])

        message = f"[Round: {str(self._round).zfill(4)}] ...{len(sampled_client_indices)} clients are selected and updated (with total sample size: {str(selected_total_size)})!"
        print(message); logging.info(message)
        del message; gc.collect()

        return selected_total_size
    
    def mp_update_selected_clients(self, selected_index):
        """Multiprocessing-applied version of "update_selected_clients" method."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected client {str(self.clients[selected_index].id).zfill(4)}...!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        self.clients[selected_index].client_update(self._round)
        client_size = len(self.clients[selected_index])
        message = f"[Round: {str(self._round).zfill(4)}] ...client {str(self.clients[selected_index].id).zfill(4)} is selected and updated (with total sample size: {str(client_size)})!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        return client_size

    def average_model(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        # average target model
        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.model.load_state_dict(averaged_weights)

        message = f"[Round: {str(self._round).zfill(4)}] ...updated weights of {len(sampled_client_indices)} clients are successfully averaged!"
        print(message); logging.info(message)
        del message; gc.collect()
    
    def evaluate_selected_models(self, sampled_client_indices):
        """Call "client_evaluate" function of each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
        print(message); logging.info(message)
        del message; gc.collect()

        for idx in sampled_client_indices:
            self.clients[idx].client_evaluate()

        message = f"[Round: {str(self._round).zfill(4)}] ...finished evaluation of {str(len(sampled_client_indices))} selected clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def mp_evaluate_selected_models(self, selected_index):
        """Multiprocessing-applied version of "evaluate_selected_models" method."""
        self.clients[selected_index].client_evaluate()
        return True


    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()

        # send global model to the selected clients
        # self.transmit_model(sampled_client_indices)
        
        self.aggregate_parameters(sampled_client_indices)
        
        for idx, client_idx in enumerate(sampled_client_indices):
            self.clients[client_idx].set_parameters(self.W_glob, self.omega, idx)        

        # updated selected clients with local dataset
        if self.mp_flag:
            with pool.ThreadPool(os.cpu_count() - 1) as workhorse:
                selected_size = workhorse.map(self.mp_update_selected_clients, sampled_client_indices)
            selected_total_size = sum(selected_size)
        else:
            selected_total_size = self.update_selected_clients(sampled_client_indices)

        # # calculate averaging coefficient of weights
        # mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]

        # # average each updated model parameters of the selected clients and update the global model
        # self.average_model(sampled_client_indices, mixing_coefficients)


    def evaluate_global_model(self):
        """Evaluate the global model using the test dataset  """
        self.model.eval()
        # self.model.to(self.device)

        acc = []; loss = []

        for testloader in self.testloader_list:

            correct, test_loss = 0, 0
            with torch.no_grad():
                for data, labels in testloader:
                    data, labels = data.float().to(self.device), labels.long().to(self.device)
                    preds = self.model(data)
                    test_loss += torch.nn.__dict__[self.model_config['criterion']]()(preds, labels).item()

                    predicted = preds.argmax(dim=1, keepdim=True)
                    correct += predicted.eq(labels.view_as(predicted)).sum().item()


            test_accuracy = correct / len(testloader.dataset)
            test_loss = test_loss / len(testloader)
            acc.append(test_accuracy)
            loss.append(test_loss)

        # self.model.to('cpu')
        # if self.device != 'cpu': torch.cuda.empty_cache()
        return acc, loss

    def fit(self):
        """Execute the whole process of the federated learning."""
        self.results = {}
        for i in range(self.num_clients):
            self.results[f"{i}"] = []

            
        def client_test(client_id):
            client_testloader = [
                self.testloader_list[int(self.num_clients*0.3)-1],
                self.testloader_list[int(self.num_clients*0.4)-1],
                self.testloader_list[int(self.num_clients*0.5)-1],
                self.testloader_list[int(self.num_clients*0.7)-1],
                self.testloader_list[int(self.num_clients*0.8)-1],
                self.testloader_list[int(self.num_clients*0.9)-1],
                self.testloader_list[int(self.num_clients*1.0)-1],
            ]
            client_testloader.append(self.testloader_list[self.num_classes+client_id])
            acc = self.clients[client_id].test(client_testloader)
            return acc
        
        with pool.ThreadPool(5) as workhorse:
            acc_list = workhorse.map(client_test, list(range(self.num_clients)))
            
        for i in range(self.num_clients):
            self.results[f"{i}"].append(acc_list[i])

        for r in range(self.num_rounds):
            self._round = r + 1
            
            self.train_federated_model()

            with pool.ThreadPool(10) as workhorse:
                acc_list = workhorse.map(client_test, list(range(self.num_clients)))
                
            for i in range(self.num_clients):
                self.results[f"{i}"].append(acc_list[i])

            message = f"[Round: {str(self._round).zfill(4)}] Evaluate global model's performance...!\
                \n\t[Server] ...finished evaluation!"

            print(message); logging.info(message)
            del message; gc.collect()

            with open(os.path.join(f"{self.log_dir}", "result.pkl"), "wb") as f:
                pickle.dump(self.results, f)

        self.transmit_model()

    def flatten(self, model):
        state_dict = model.state_dict()
        keys = state_dict.keys()
        W = [state_dict[key].flatten() for key in keys]
        return torch.cat(W)
    
    def aggregate_parameters(self, sampled_client_indices):
        self.W_glob = torch.zeros((self.dim, self.join_clients), device=self.device)
        for idx, client_idx in enumerate(sampled_client_indices):
            self.W_glob[:, idx] = self.flatten(self.clients[client_idx].model)