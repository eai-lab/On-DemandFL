from ast import arg
from inspect import stack
import os
import sys
import time
import datetime
import pickle
import threading
import logging
import numpy as np

from src.set_seed import set_seed
from src.options import args_parser

from torch.utils.tensorboard import SummaryWriter

from src.server import Server
from src.utils import launch_tensor_board
from src.sampling import prepare_dataset

def softmax(arr):
    return arr/np.sum(arr)


if __name__ == "__main__":
    time_config = str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    
    args = args_parser()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    set_seed(args.seed)

    if args.dataset_name.upper() == 'MNIST':
        num_classes = 10
        model_name = 'mnist_fc'
    if args.dataset_name.upper() == 'FASHIONMNIST':
        num_classes = 10
        model_name = 'fashionmnist_cnn'
    elif args.dataset_name.upper() == 'CIFAR10':
        num_classes = 10
        model_name = 'cifar10_cnn'
    elif args.dataset_name.upper() == 'SVHN':
        num_classes = 10
        model_name = 'svhn_cnn'
    elif args.dataset_name.upper() == 'CIFAR100':
        num_classes = 100
        model_name = 'cifar100_cnn'
    elif args.dataset_name.upper() == "EMNIST":
        num_classes = 62
        model_name = 'emnist_cnn'

    target_dist_list = []
    # add 100% 90%, 80%, 70%, 60%, 50%, 30%, 20%, 10% 
    label_distribution = np.random.default_rng(seed=args.seed).dirichlet(alpha=np.repeat(args.alpha, args.num_clients), size=num_classes)
    for dist in label_distribution.T:
        target_dist_list.append(softmax(dist))

    log_dir = f'{args.log_dir}/{args.dataset_name}'
    log_path, log_file = log_dir, 'FL_Log.log'
    log_path = os.path.join(log_path, f'a_{args.alpha}')

    fed_config = {
        'num_rounds': args.rounds, 'num_clients': args.num_clients, 'fraction': args.fraction, 'alpha': args.alpha
    }
    data_config = {
        'name': args.dataset_name, 'num_classes': num_classes, 'alpha': args.alpha
    }
    task_model_config = {
        'lr': args.tm_lr, 'momentum': args.tm_momentum, 'name': model_name,
        'criterion': args.tm_criterion, 'optimizer': args.tm_optimizer,
        'ep': args.tm_local_ep, 'bs': args.tm_local_bs
    }
    system_config = {
        'is_mp': args.mp, 'device': 'cuda', 'seed': args.seed, 'log_dir': log_path, 'time_config': time_config
    }

    # initiate TensorBaord for tracking losses and metrics
    writer = SummaryWriter(log_dir=log_path, filename_suffix="FL")

    # set the configuration of global logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(log_path, log_file),
        level=logging.INFO,
        format="[%(levelname)s](%(asctime)s) %(message)s",
        datefmt="%Y/%m/%d/ %I:%M:%S %p")

    logging.info(f"fed_config: {fed_config}")
    logging.info(f"task_model_config: {task_model_config}")
    logging.info(f"system_config: {system_config}")

    message = f"\n[Target Distribution] ${target_dist_list}"; logging.info(message)

    # federated learning
    partitioned_train_set, test_dataset_list = prepare_dataset(seed=args.seed, dataset_name=args.dataset_name, num_client=args.num_clients, 
                                                            alpha=args.alpha, target_dist_list=target_dist_list)
    message = "\n[WELCOME] Unfolding configurations...!"
    print(message); logging.info(message)

    central_server = Server(writer, partitioned_train_set, test_dataset_list,
                            fed_config, data_config, task_model_config, system_config)
    central_server.setup()
    central_server.fit()

    with open(os.path.join(log_path, "result.pkl"), "wb") as f:
        pickle.dump(central_server.results, f)

    message = "...done all learning process!\n...exit program!"
    print(message); logging.info(message)
    time.sleep(3); os._exit(0)
  
