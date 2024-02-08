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

import wandb

logger = logging.getLogger(__name__)


def softmax(x):
    return np.array(x) / sum(x)

class Server(object):
    """Class for implementing center server orchestrating the whole process of federated learning
    
    At first, center server distribute model skeleton to all participating clients with configurations.
    While proceeding federated learning rounds, the center server samples some fraction of clients,
    receives locally updated parameters, averages them as a global parameter (model), and apply them to global model.
    In the next round, newly selected clients will recevie the updated global model as its local model.  

    """

    def __init__(self, train_dataset, test_dataset_list, arguments):
        self.arguments = arguments
        self.clients = None
        self._round = 0

        # dataset 
        self.num_classes = int(arguments['num_classes'])
        self.dataset_name = arguments['dataset_name']
        self.alpha = arguments['alpha']
        self.train_dataset = train_dataset
        self.test_dataset_list = test_dataset_list
        self.target_dist_op = arguments['target_dist_op']
        self.target_dist = softmax(self.gen_label_dist_from_dataset(test_dataset_list[self.target_dist_op]))
        
        # system setting
        self.device = arguments['device']
        self.mp_flag = arguments['is_mp']
        self.log_dir = arguments['log_dir']
        self.baseline_dir_name = arguments['baseline_dir_name']
        self.wandb = arguments['wandb']

        # federated setting
        self.num_clients = arguments['num_clients']
        self.num_rounds = arguments['num_rounds']
        self.fraction = arguments['fraction']
        self.init_round = arguments['init_round']
        self.check_point = arguments['check_point']
        self.method = arguments['method']
        self.subset_size = arguments['subset_size']
       
        self.dm_model_idx = arguments['dm_model_idx']
        self.dm_pred_method = arguments['dm_pred_method']
        
        # task model
        self.tm_criterion = arguments['tm_criterion']        
        
        # load task model
        self.task_model = eval(arguments['tm_name'])().to(self.device)
        if os.path.exists(f"{self.log_dir}/check_point.pkl"):
            with open(f"{self.log_dir}/check_point.pkl", 'rb') as f:
                self.check_point = pickle.load(f)
            task_model_path = f"{self.log_dir}/task_model.pt"
            self.task_model.load_state_dict(torch.load(task_model_path))
        else:
            task_model_path = f"../baseline/log/{self.dataset_name}/{self.baseline_dir_name}/models/{self.dataset_name}-a{self.alpha}-{self.init_round}-task_model.pt"
            self.task_model.load_state_dict(torch.load(task_model_path))
            self.check_point = 0
            
        self.task_model = self.task_model.to(self.device)
        
        message = f"{self.task_model}\n"
        print(message); logging.info(message)
        del message; gc.collect()
    
    def select_subset(self):
        # selec sub clients set
        client_dist = self.client_dist.copy()
        total_datasize = 0
        for client in self.clients:
            total_datasize += len(client)

        for i in range(self.num_clients):
            client_dist[i] = client_dist[i] * (len(self.clients[i])/total_datasize)

        selected_client_subset = set()   
        recursive_weight = [0]*self.num_clients
        # subset_size = max(int(self.fraction * self.num_clients), 1)
        subset_size = self.subset_size

        while (len(selected_client_subset) < subset_size):
            preds_w, preds_err = nnls(client_dist.T, self.target_dist)

            for i in np.where(preds_w != 0)[0]:
                selected_client_subset.add(i)
                recursive_weight[i] += preds_w[i]
                client_dist[i] = np.array([0]*client_dist[i].shape[0])

        recursive_weight = softmax(np.array(recursive_weight)).flatten()

        self.weight = recursive_weight
        self.client_subset = np.where(recursive_weight != 0)[0]

        approximated_target_dist = self.weight.dot(self.client_dist)
        mae_error = np.mean(np.abs(self.target_dist - approximated_target_dist))

        message = f"[Subset Select] selected subset is {self.client_subset}, subset size is {len(self.client_subset)} and weight is {self.weight}\
                    \n approximated target distribution is {approximated_target_dist}\
                    \n MAE (approximated target distribution and target distribution is {mae_error}"
        print(message); logging.info(message)
        del message; gc.collect()

    def gen_label_dist_from_dataset(self, dataset):
        idcs = np.array([i for i in range(0, len(dataset.data))])
        labels = np.array(dataset.targets, dtype=np.int32)
        class_idcs = [np.argwhere(labels[idcs]==y).flatten() for y in range(self.num_classes)]
        
        label_dist = [len(class_idcs[i]) for i in range(self.num_classes)]
        print(label_dist)
        return label_dist

    def setup(self, **init_kwargs):
        """Set up all configuration for federated learning."""
        # valid only before the very first round
        assert self._round == 0

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.task_model.parameters()))})!"
        print(message); logging.info(message)
        del message; gc.collect()

        # assign dataset to each client
        self.clients = self.create_clients(self.train_dataset)

        result_path = f"../baseline/log/{self.dataset_name}/{self.baseline_dir_name}/result.pkl"        
        with open(result_path, 'rb') as f:
            result = pickle.load(f)
            
        # self.client_dist = np.array([pred.numpy() for pred in result[self.dm_pred_method][self.init_round][self.dm_model_idx]])
        
        self.client_dist = []
        for c in self.clients:
            self.client_dist.append(c.dist)
        self.client_dist = np.concatenate(self.client_dist, axis=0)
        print(self.client_dist.shape)

        
        # prepare hold-out dataset for evaluation
        self.testloader_list = [
            DataLoader(testset, batch_size=256, shuffle=False)
            for testset in self.test_dataset_list
        ]

        # configure detailed settings for client upate and 
        self.setup_clients(
            self.arguments
        )
        
        # select subset
        self.select_subset()
        
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

    def setup_clients(self, arguments):
        """Set up each client."""
        for k, client in tqdm(enumerate(self.clients), leave=False):
            client.setup(arguments)
        
        message = f"[Round: {str(self._round).zfill(4)}] ...successfully finished setup of all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def transmit_model(self, sampled_client_indices=None):
        """Send the updated global model to selected/all clients."""
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            assert (self._round == 0) or (self._round == self.num_rounds)

            for client in tqdm(self.clients, leave=False):
                client.task_model = copy.deepcopy(self.task_model)

            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
            print(message); logging.info(message)
            del message; gc.collect()
        else:
            # send the global model to selected clients
            assert self._round != 0

            for idx in tqdm(sampled_client_indices, leave=False):
                self.clients[idx].task_model = copy.deepcopy(self.task_model)
            
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
        d_selected_total_size = 0
        for idx in tqdm(sampled_client_indices, leave=False):
            self.clients[idx].client_update(self._round)
            selected_total_size += len(self.clients[idx])
            d_selected_total_size += len(self.clients[idx].task_model_param_list)

        message = f"[Round: {str(self._round).zfill(4)}] ...{len(sampled_client_indices)} clients are selected and updated (with total sample size: {str(selected_total_size)}, {str(d_selected_total_size)})!"
        print(message); logging.info(message)
        del message; gc.collect()

        return selected_total_size, d_selected_total_size
    
    def mp_update_selected_clients(self, selected_index):
        """Multiprocessing-applied version of "update_selected_clients" method."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected client {str(self.clients[selected_index].id).zfill(4)}...!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        self.clients[selected_index].client_update(self._round)
        client_size = len(self.clients[selected_index])
        # d_client_size = len(self.clients[selected_index].task_model_param_list)

        message = f"[Round: {str(self._round).zfill(4)}] ...client {str(self.clients[selected_index].id).zfill(4)} is selected and updated (with total sample size: {str(client_size)})!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        return client_size

    def average_model(self, sampled_client_indices, coefficients, weight_efficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        # average target model
        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].task_model.state_dict()
            for key in self.task_model.state_dict().keys():
                if self.method == 1:
                    if it == 0:
                        averaged_weights[key] = coefficients[it] * local_weights[key]
                    else:
                        averaged_weights[key] += coefficients[it] * local_weights[key]
                elif self.method == 2:
                    if it == 0:
                        averaged_weights[key] = weight_efficients[it] * local_weights[key]
                    else:
                        averaged_weights[key] += weight_efficients[it] * local_weights[key]
                    
        self.task_model.load_state_dict(averaged_weights)

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
    
    def select_subset_client(self):
        """Select some fraction of all clients."""
        message = f"[Round: {str(self._round).zfill(4)}] Select clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        if len(self.client_subset) <= num_sampled_clients:
            return self.client_subset
        else:
            if self.method == 1:
                sampled_client_indices = sorted(np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients, p=self.weight, replace=False).tolist())
            elif self.method == 2:
                sampled_client_indices = sorted(np.random.choice(a=[i for i in self.client_subset], size=num_sampled_clients, replace=False).tolist())
            return sampled_client_indices

    def update_client_model_param_list(self, sampled_client_indices):
        self.client_model_param_list = []
        self.client_dist_gt_list = []

        for c_i in sampled_client_indices:
            model_param_list = self.clients[c_i].task_model_param_list
            model_param = sum(model_param_list)/len(model_param_list)
            model_param = model_param.reshape(1, -1)
            dist_gt = self.clients[c_i].dist.to(self.device)

            self.client_model_param_list.append(model_param)
            self.client_dist_gt_list.append(dist_gt)

        if self._round == 1:
            self.first_client_model_param_list = [model_param for model_param in self.client_model_param_list]
            self.first_client_dist_gt_list = [dist_gt for dist_gt in self.client_dist_gt_list]

    def evaluate_dist_model_with_first_model_param(self, d_model):
        d_model.eval()
        # d_model.to(self.device)

        mae = 0
        with torch.no_grad():
            for model_param, dist_gt in zip(self.first_client_model_param_list, self.first_client_dist_gt_list):
                model_param = model_param.to(self.device)
                dist_gt = dist_gt.to(self.device)

                pred = d_model(model_param)
                mae += torch.nn.L1Loss()(pred, dist_gt).item()

        # if self.device != 'cpu': torch.cuda.empty_cache()
        # d_model.to('cpu')
        mae /= len(self.first_client_model_param_list)

        return mae

    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        # sampled_client_indices = self.sample_clients()
        sampled_client_indices = self.select_subset_client()

        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)

        # updated selected clients with local dataset
        if self.mp_flag:
            with pool.ThreadPool(10) as workhorse:
                selected_size = workhorse.map(self.mp_update_selected_clients, sampled_client_indices)
            selected_total_size = sum(selected_size)
        else:
            selected_total_size = self.update_selected_clients(sampled_client_indices)

        # calculate averaging coefficient of weights
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]


        # calculate averaging coefficient of weights
        total_weight = 0
        for idx in sampled_client_indices:
            total_weight += self.weight[idx]
        weight_efficients = softmax(np.array([self.weight[idx] / total_weight for idx in sampled_client_indices]))

        # average each updated model parameters of the selected clients and update the global model
        self.average_model(sampled_client_indices, mixing_coefficients, weight_efficients)

    def evaluate_global_task_model(self):
        """Evaluate the global model using the test dataset  """
        self.task_model.eval()
        # self.task_model.to(self.device)

        acc = []; loss = []

        correct, test_loss = 0, 0
        with torch.no_grad():
            for data, labels in self.testloader_list[self.target_dist_op]:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                preds = self.task_model(data)
                test_loss += torch.nn.__dict__[self.tm_criterion]()(preds, labels).item()

                predicted = preds.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

        test_accuracy = correct / len(self.testloader_list[self.target_dist_op].dataset)
        test_loss = test_loss / len(self.testloader_list[self.target_dist_op])

        return test_accuracy, test_loss

    
    def evaluate_global_dist_predictor(self):
        """ test with client distribution set """

        avg_dist_mae_list = []
        avg_grad_mae_list = []
        avg_dist_pred_list = []
        avg_grad_pred_list = []

        for model in self.dist_predictor_list:

            avg_dist_mae = 0;       avg_grad_mae = 0
            avg_dist_preds = None;  avg_grad_preds = None; 

            model.eval()
            # model.to(self.device)
            criterion = torch.nn.L1Loss()

            for client in self.clients:
                if client.task_model_param_list != None:
                    gradient = client.task_model_param_list.clone()
                    gradient_mean = torch.mean(gradient, 0).reshape(1, -1)
                    dist_gt = client.dist.clone()
                    
                    gradient = gradient.to(self.device)
                    gradient_mean = gradient_mean.to(self.device)
                    dist_gt = dist_gt.to(self.device)
                
                    with torch.no_grad():
                        avg_dist_pred = model(gradient)
                        avg_dist_pred = torch.mean(avg_dist_pred, 0).reshape(1, -1)
                        avg_grad_pred = model(gradient_mean)

                        avg_dist_mae += criterion(avg_dist_pred, client.dist.to(self.device)).item()
                        avg_grad_mae += criterion(avg_grad_pred, client.dist.to(self.device)).item()

                        if avg_dist_preds == None:  avg_dist_preds = avg_dist_pred.cpu().detach()
                        else:   avg_dist_preds = torch.cat([avg_dist_preds, avg_dist_pred.cpu().detach()], 0)
                        if avg_grad_preds == None:  avg_grad_preds = avg_grad_pred.cpu().detach()
                        else:   avg_grad_preds = torch.cat([avg_grad_preds, avg_grad_pred.cpu().detach()], 0)
                else:
                    if avg_dist_preds == None:  avg_dist_preds = torch.Tensor([1/self.num_classes]*self.num_classes).reshape(1, -1)
                    else:   avg_dist_preds = torch.cat([avg_dist_preds, torch.Tensor([1/self.num_classes]*self.num_classes).reshape(1, -1)], 0)
                    if avg_grad_preds == None:  avg_grad_preds = torch.Tensor([1/self.num_classes]*self.num_classes).reshape(1, -1)
                    else:   avg_grad_preds = torch.cat([avg_grad_preds, torch.Tensor([1/self.num_classes]*self.num_classes).reshape(1, -1)], 0)
            if self.device != 'cpu': torch.cuda.empty_cache()
            # model.to('cpu')

            avg_dist_mae /= self.num_clients
            avg_grad_mae /= self.num_clients
            
            avg_dist_mae_list.append(avg_dist_mae)
            avg_grad_mae_list.append(avg_grad_mae)
            avg_dist_pred_list.append(avg_dist_preds)
            avg_grad_pred_list.append(avg_grad_preds)

        return avg_dist_mae_list, avg_grad_mae_list, avg_dist_pred_list, avg_grad_pred_list


    def fit(self):
        """Execute the whole process of the federated learning."""
        if self.check_point == 0:
            self.results = {"loss": [], "accuracy": []} 
            test_accuracy, test_loss = self.evaluate_global_task_model()
            self.results['loss'].append(test_loss)
            self.results['accuracy'].append(test_accuracy)
        else:
            with open(os.path.join(f"{self.log_dir}", "result.pkl"), 'rb') as f:
                self.results = pickle.load(f)
                
        for r in range(self.check_point, self.num_rounds):
            self._round = r + 1
            
            self.train_federated_model()

            test_accuracy, test_loss = self.evaluate_global_task_model()
            self.results['loss'].append(test_loss)
            self.results['accuracy'].append(test_accuracy)
            
            if self.wandb:
                wandb.log({f'GT-OnDemFL-{self.target_dist_op}': test_accuracy, 'round': self.init_round + self.check_point + self._round})

            message = f"[Round: {str(self._round).zfill(4)}] Evaluate global model's performance...!\
                \n\t[Server] ...finished evaluation!\
                \n\t=> Loss: {test_loss}\
                \n\t=> Accuracy: {test_accuracy}"
            print(message); logging.info(message)
            del message; gc.collect()
            
            with open(os.path.join(f"{self.log_dir}", "result.pkl"), "wb") as f:
                pickle.dump(self.results, f)

            with open(os.path.join(f"{self.log_dir}", "check_point.pkl"), "wb") as f:
                pickle.dump(self._round, f)
                
                
            torch.save(self.task_model.state_dict(), f"{self.log_dir}/task_model.pt")
            
        self.transmit_model()
                