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
        self.train_dataset = train_dataset
        self.test_dataset_list = test_dataset_list
        self.num_classes = int(arguments['num_classes'])
        self.dataset_name = arguments['dataset_name']
        self.alpha = arguments['alpha']
        
        # system setting
        self.device = arguments['device']
        self.mp_flag = arguments['is_mp']
        self.log_dir = arguments['log_dir']
        self.wandb = arguments['wandb']

        # federated setting
        self.num_clients = arguments['num_clients']
        self.num_rounds = arguments['num_rounds']
        self.fraction = arguments['fraction']
        self.init_round = arguments['init_round']
        
        # task model
        self.tm_criterion = arguments['tm_criterion']        
        
        # gen model
        self.task_model = eval(arguments['tm_name'])().to(self.device)

        # test 5 distribution predictor
        self.dist_predictor_list = [
            eval('distribution_fcn')(input=arguments['dm_input_size'], output=arguments['dm_output_size']).to(self.device),
            # eval('distribution_fcn2')(input=arguments['dm_input_size'], output=arguments['dm_output_size']).to(self.device),
            # eval('distribution_fcn5')(input=arguments['dm_input_size'], output=arguments['dm_output_size']).to(self.device),
        ]
        # load model
        if self.init_round == None:
            self.init_round = 0
        else:
            self.task_model.load_state_dict(torch.load(f"{self.log_dir}/models/{self.dataset_name}-a{self.alpha}-{self.init_round}-task_model.pt"))
            for i in range(len(self.dist_predictor_list)):
                self.dist_predictor_list[i].load_state_dict(torch.load(f"{self.log_dir}/models/{self.dataset_name}-a{self.alpha}-{self.init_round}-dist_predictor_{i+1}.pt"))
        
        message = f"{self.task_model}\n"
        print(message); logging.info(message)
        del message; gc.collect()

        self.max_acc = 0
        self.max_acc_round = 0
        self.save_upper_limit = arguments['save_upper_limit']
        
        
    def setup(self, **init_kwargs):
        """Set up all configuration for federated learning."""
        # valid only before the very first round
        assert self._round == 0

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.task_model.parameters()))})!"
        print(message); logging.info(message)
        del message; gc.collect()

        # assign dataset to each client
        self.clients = self.create_clients(self.train_dataset)
        
        # save client's distribution for debuging
        self.client_dist = []
        for c in self.clients:
            self.client_dist.append(c.dist)
        with open(f"./{self.log_dir}/{self.dataset_name}-a{self.alpha}-client_distribution.pkl", 'wb') as f:
            pickle.dump(self.client_dist, f)

        # prepare hold-out dataset for evaluation
        self.testloader_list = [
            DataLoader(testset, batch_size=256, shuffle=False)
            for testset in self.test_dataset_list
        ]

        # configure detailed settings for client upate and 
        self.setup_clients(
            self.arguments
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
                client.dist_predictor_list = copy.deepcopy(self.dist_predictor_list)

            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
            print(message); logging.info(message)
            del message; gc.collect()
        else:
            # send the global model to selected clients
            assert self._round != 0

            for idx in tqdm(sampled_client_indices, leave=False):
                self.clients[idx].task_model = copy.deepcopy(self.task_model)
                self.clients[idx].dist_predictor_list = copy.deepcopy(self.dist_predictor_list)
            
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
        d_client_size = len(self.clients[selected_index].task_model_param_list)

        message = f"[Round: {str(self._round).zfill(4)}] ...client {str(self.clients[selected_index].id).zfill(4)} is selected and updated (with total sample size: {str(client_size)})!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        return client_size, d_client_size

    def average_model(self, sampled_client_indices, coefficients, d_coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        # average target model
        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].task_model.state_dict()
            for key in self.task_model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.task_model.load_state_dict(averaged_weights)

        # average distribution model
        for i in range(len(self.dist_predictor_list)):
            averaged_weights = OrderedDict()
            for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
                local_weights = self.clients[idx].dist_predictor_list[i].state_dict()
                for key in self.dist_predictor_list[i].state_dict().keys():
                    if it == 0:
                        averaged_weights[key] = d_coefficients[it] * local_weights[key]
                    else:
                        averaged_weights[key] += d_coefficients[it] * local_weights[key]
            self.dist_predictor_list[i].load_state_dict(averaged_weights)

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
        sampled_client_indices = self.sample_clients()

        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)

        # updated selected clients with local dataset
        if self.mp_flag:
            with pool.ThreadPool(10) as workhorse:
                selected_size = workhorse.map(self.mp_update_selected_clients, sampled_client_indices)
            selected_total_size, d_selected_total_size = 0, 0
            for i in selected_size:
                selected_total_size += i[0]
                d_selected_total_size += i[1]
        else:
            selected_total_size, d_selected_total_size = self.update_selected_clients(sampled_client_indices)

        # calculate averaging coefficient of weights
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]
        d_mixing_coefficients = [len(self.clients[idx].task_model_param_list) / d_selected_total_size for idx in sampled_client_indices]

        # average each updated model parameters of the selected clients and update the global model
        self.average_model(sampled_client_indices, mixing_coefficients, d_mixing_coefficients)
        
        # update clients model parameter list for predict their distribution
        # self.update_client_model_param_list(sampled_client_indices)


    def evaluate_global_task_model(self):
        """Evaluate the global model using the test dataset  """
        self.task_model.eval()
        # self.task_model.to(self.device)

        acc = []; loss = []

        for testloader in self.testloader_list:

            correct, test_loss = 0, 0
            with torch.no_grad():
                for data, labels in testloader:
                    data, labels = data.float().to(self.device), labels.long().to(self.device)
                    preds = self.task_model(data)
                    test_loss += torch.nn.__dict__[self.tm_criterion]()(preds, labels).item()

                    predicted = preds.argmax(dim=1, keepdim=True)
                    correct += predicted.eq(labels.view_as(predicted)).sum().item()


            test_accuracy = correct / len(testloader.dataset)
            test_loss = test_loss / len(testloader)
            acc.append(test_accuracy)
            loss.append(test_loss)

        # self.task_model.to('cpu')
        # if self.device != 'cpu': torch.cuda.empty_cache()
        return acc, loss

    
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
        if self.init_round == 0 or self.init_round == None:
            self.results = {"loss": [], "accuracy": [], "avg_dist_mae": [], "avg_grad_mae": [], "avg_dist_pred": [], "avg_grad_pred": []}
        else:
            with open(os.path.join(f"{self.log_dir}", "result.pkl"), "rb") as f:
                self.results = pickle.load(f)
                
        model_save_path = f"{self.log_dir}/models/"
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
                    
        # save non-trained model                    
        torch.save(self.task_model.state_dict(), f"{model_save_path}/{self.dataset_name}-a{self.alpha}-{self._round}-task_model.pt")
  
        test_accuracy, test_loss = self.evaluate_global_task_model()
        avg_dist_mae_list, avg_grad_mae_list, avg_dist_pred_list, avg_grad_preds_list = self.evaluate_global_dist_predictor()
        self.results['loss'].append(test_loss)
        self.results['accuracy'].append(test_accuracy)
        self.results['avg_dist_mae'].append(avg_dist_mae_list)
        self.results['avg_grad_mae'].append(avg_grad_mae_list)
        self.results['avg_dist_pred'].append(avg_dist_pred_list)
        self.results['avg_grad_pred'].append(avg_grad_preds_list)
        
        if self.wandb:
            for i, acc in enumerate(test_accuracy):
                wandb.log({f'FedAvg-{i}': acc, 'round': 0})

        for r in range(self.init_round, self.num_rounds):
            self._round = r + 1
            
            self.train_federated_model()

            test_accuracy, test_loss = self.evaluate_global_task_model()
            avg_dist_mae_list, avg_grad_mae_list, avg_dist_pred_list, avg_grad_preds_list = self.evaluate_global_dist_predictor()
            self.results['loss'].append(test_loss)
            self.results['accuracy'].append(test_accuracy)
            self.results['avg_dist_mae'].append(avg_dist_mae_list)
            self.results['avg_grad_mae'].append(avg_grad_mae_list)
            self.results['avg_dist_pred'].append(avg_dist_pred_list)
            self.results['avg_grad_pred'].append(avg_grad_preds_list)
                
            if self.wandb:
                for i, acc in enumerate(test_accuracy):
                    wandb.log({f'FedAvg-{i}': acc, 'round': self._round})
            if self.wandb:
                for i, mae in enumerate(avg_dist_mae_list):
                    wandb.log({f'{i}-MAE-dist': mae, 'round': self._round})
            
            if self._round%50 == 0:
                self.max_acc = 0
            
            # check base model with testset (original testset)
            if test_accuracy[0] > self.max_acc and self._round <= self.save_upper_limit:
                upper_limit = (self._round//50 + 1)*50
                self.max_acc_round = self._round
                self.max_acc = test_accuracy[0]
                torch.save(self.task_model.state_dict(), f"{model_save_path}/{self.dataset_name}-a{self.alpha}-best-{upper_limit}-task_model.pt")

                with open(os.path.join(f"{self.log_dir}", f"best-model-{upper_limit}-info.txt"), "wb") as f:
                    message = f"\n\nround %d \nmax acc %f" %(self._round, self.max_acc)
                    f.write(message.encode('utf-8'))                    
            
            message = f"[Round: {str(self._round).zfill(4)}] Evaluate global model's performance...!\
                \n\t[Server] ...finished evaluation!\
                \n\t=> Loss: {test_loss}\
                \n\t=> Accuracy: {test_accuracy}\
                \n\t=> Accuracy (Max): {self.max_acc}\
                \n\t=> Accuracy (Max) Round: {self.max_acc_round}\
                \n\t=> MAE (dist avg): {avg_dist_mae_list}\
                \n\t=> MAE (grad avg): {avg_grad_mae_list}"
            print(message); logging.info(message)
            del message; gc.collect()

                        
            if (self._round % 50 == 0) or (self._round == 20):   
                torch.save(self.task_model.state_dict(), f"{model_save_path}/{self.dataset_name}-a{self.alpha}-{self._round}-task_model.pt")
                            
                for i in range(0, len(self.dist_predictor_list)):
                    torch.save(self.dist_predictor_list[i].state_dict(), f"{model_save_path}/{self.dataset_name}-a{self.alpha}-{self._round}-dist_predictor_{i+1}.pt")
                    
            with open(os.path.join(f"{self.log_dir}", "result.pkl"), "wb") as f:
                pickle.dump(self.results, f)

        self.transmit_model()

