from ast import arg
from inspect import stack
import os, sys, time, datetime, threading
import pickle, logging, wandb
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
    if args.device != 'cpu':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        device = 'cuda'
    else:
        device = 'cpu'

    set_seed(args.seed)

    if args.dataset_name.upper() == 'MNIST':
        num_classes = 10
        tm_name = 'mnist_fc'
        dm_input_size = 320
        save_upper_limit=100
    if args.dataset_name.upper() == 'FASHIONMNIST':
        num_classes = 10
        tm_name = 'fashionmnist_cnn'
        dm_input_size = 5120
        save_upper_limit=300
    elif args.dataset_name.upper() == 'CIFAR10':
        num_classes = 10
        tm_name = 'cifar10_cnn'
        dm_input_size = 5120
        save_upper_limit=300
    elif args.dataset_name.upper() == 'SVHN':
        num_classes = 10
        tm_name = 'svhn_cnn'
        dm_input_size = 5120
        save_upper_limit=300
        # dm_input_size = num_classes if args.dm_input_type == 'y_derivative' else 5120
    elif args.dataset_name.upper() == 'CIFAR100':
        num_classes = 100
        tm_name = 'cifar100_cnn'
        dm_input_size = 128000
        save_upper_limit=1500
        # dm_input_size = num_classes if args.dm_input_type == 'y_derivative' else 128000
    elif args.dataset_name.upper() == "EMNIST":
        num_classes = 62
        tm_name = 'emnist_cnn'
        dm_input_size = 126976
        save_upper_limit=1000
        # dm_input_size = num_classes if args.dm_input_type == 'y_derivative' else 126976


    target_dist_list = []
    # add 100% 90%, 80%, 70%, 60%, 50%, 30%, 20%, 10% 
    label_distribution = np.random.default_rng(seed=args.seed).dirichlet(alpha=np.repeat(args.alpha, args.num_clients), size=num_classes)
    for dist in label_distribution.T:
        target_dist_list.append(softmax(dist))
            
    log_dir = f'{args.log_dir}/{args.dataset_name}'
    log_path, log_file = log_dir, 'FL_Log.log'
    dir_name = f'nc_{args.num_clients}-a_{args.alpha}-tep_{args.tm_local_ep}-tbs_{args.tm_local_bs}-dep_{args.dm_local_ep}-dbs_{args.dm_local_bs}-dmc_{args.dm_criterion}'
    log_path = os.path.join(log_path, dir_name)

    fed_config = {
        'num_rounds': args.num_rounds, 'num_clients': args.num_clients, 'fraction': args.fraction, 'alpha': args.alpha, 
        'save_upper_limit': save_upper_limit, 'init_round': args.init_round
    }
    data_config = {
        'name': args.dataset_name, 'num_classes': num_classes, 'alpha': args.alpha, 
    }
    task_model_config = {
        'lr': args.tm_lr, 'momentum': args.tm_momentum, 'name': tm_name,
        'criterion': args.tm_criterion, 'optimizer': args.tm_optimizer,
        'ep': args.tm_local_ep, 'bs': args.tm_local_bs
    }
    dist_predictor_config = {
        'lr': args.dm_lr, 'name': args.dm_name, 'input_size': dm_input_size, 'output_size': num_classes, # 'input_type': args.dm_input_type, 
        'criterion': args.dm_criterion, 'optimizer': args.dm_optimizer,
        'ep': args.dm_local_ep, 'bs': args.dm_local_bs,
    }
    system_config = {
        'is_mp': args.mp, 'device': device, 'seed': args.seed, 'log_dir': log_path, 'time_config': time_config
    }
    arguments = {
        'num_rounds': args.num_rounds,
        'num_clients': args.num_clients,
        'fraction': args.fraction,
        'alpha': args.alpha,
        'save_upper_limit': save_upper_limit,
        'init_round': args.init_round,
        
        'dataset_name': args.dataset_name,
        'num_classes': num_classes,
        'alpha': args.alpha,
        
        'tm_lr': args.tm_lr,
        'tm_momentum': args.tm_momentum,
        'tm_name': tm_name,
        'tm_criterion': args.tm_criterion,
        'tm_optimizer': args.tm_optimizer,
        'tm_ep': args.tm_local_ep,
        'tm_bs': args.tm_local_bs,
        
        'dm_lr': args.dm_lr,
        'dm_name': args.dm_name,
        'dm_input_size': dm_input_size,
        'dm_output_size': num_classes,
        'dm_criterion': args.dm_criterion,
        'dm_momentum': args.dm_momentum,
        
        'dm_optimizer': args.dm_optimizer,
        'dm_ep': args.dm_local_ep,
        'dm_bs': args.dm_local_bs,
        
        'is_mp': args.mp,
        'device': device,
        'seed': args.seed,
        'log_dir': log_path,
        'time_config': time_config,
        'wandb': args.wandb
    }
    if args.wandb == True:
        wandb.init(project = args.wandb_project_name, #'on-demand-fl-distribution-predictor', 
                   config = arguments, 
                   name = f'{dir_name}')
    

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
    logging.info(f"dist_predictor_config: {dist_predictor_config}")
    logging.info(f"system_config: {system_config}")


    # federated learning
    partitioned_train_set, test_dataset_list = prepare_dataset(seed=args.seed, dataset_name=args.dataset_name, num_client=args.num_clients, 
                                                            alpha=args.alpha, target_dist_list=target_dist_list)
    message = "\n[WELCOME] Unfolding configurations...!"
    print(message); logging.info(message)

    central_server = Server(partitioned_train_set, test_dataset_list, arguments)
    central_server.setup()
    central_server.fit()

    with open(os.path.join(log_path, "result.pkl"), "wb") as f:
        pickle.dump(central_server.results, f)

    message = "...done all learning process!\n...exit program!"
    print(message); logging.info(message)
    time.sleep(3); os._exit(0)
  
