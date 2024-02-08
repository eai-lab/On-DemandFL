import argparse

def str_to_bool(param):
    if isinstance(param, bool):
        return param
    if param.lower() in ('true', '1'): 
        return True
    elif param.lower() in ('false', '0'):
        return False
    else:
        raise argparse.argparse.ArgumentTypeError('boolean value expected')

def args_parser():
    parser = argparse.ArgumentParser()

    # federated learning arguments
    parser.add_argument('--num_rounds', type=int, default=100, help="number of round of training")
    parser.add_argument('--num_clients', type=int, default=100, help='number of client (K)')
    parser.add_argument('--fraction', type=float, default=0.1, help='fraction of client (C)')
    parser.add_argument('--tm_local_ep', type=int, default=1, help='the number of local epochs of task model: E_t')
    parser.add_argument('--dm_local_ep', type=int, default=1, help='the number of local epochs of distribution model: E_d')
    parser.add_argument('--tm_local_bs', type=int, default=10, help='batch size of local target model: B_t')
    parser.add_argument('--dm_local_bs', type=int, default=10, help='batch size of local distribution model: B_t')
    parser.add_argument('--init_round', type=int)
    parser.add_argument('--client_pred_round', type=int)
    parser.add_argument('--check_point', type=int)
    parser.add_argument('--method', type=int, help='aggregation method and subset selecting method')
    parser.add_argument('--subset_size', type=int)
    parser.add_argument('--dm_model_idx', type=int)
    parser.add_argument('--dm_pred_method')

    # target model arguments
    parser.add_argument('--tm_lr', type=float, default=0.01, help='learning rate of task model')
    parser.add_argument('--tm_criterion', default='CrossEntropyLoss', help='criterion of task model')
    parser.add_argument('--tm_optimizer', default='SGD', help='optimizer of task model')
    parser.add_argument('--tm_momentum', type=float, default=0.9, help='momentum of optimizer of task model')
    # parser.add_argument('--tm_name', help='task model name')
    # parser.add_argument('--tm_num_classes', type=int, help='number of classes of dataset')

    # distribution model arguments
    parser.add_argument('--dm_lr', type=float, default=0.001, help='learning rate of task model')
    parser.add_argument('--dm_criterion', default='MultiLabelSoftMarginLoss', help='criterion of distribution model')
    # parser.add_argument('--dm_optimizer', default='Adam', help='optimizer of distribution model')
    parser.add_argument('--dm_optimizer', default='SGD', help='optimizer of distribution model')
    parser.add_argument('--dm_name', default='distribution_fcn', help='task model name')
    parser.add_argument('--dm_momentum', default=0.5)
    # parser.add_argument('--dm_input_size', type=int, help='size of input of distribution model')
    # parser.add_argument('--dm_preprocessing', default='none', help='model parameter preprocessing option: none/norm_normalization/min_max')
    # parser.add_argument('--dm_input_type', default='gradient', help='gradient/weight/y_derivative/weighted_average')

    # dataset setting
    parser.add_argument('--dataset_name', help='dataset name')
    parser.add_argument('--alpha',  type=float, default=1, help='alpha of dirichlet distribution')
    parser.add_argument('--target_dist_op', type=int, default=4, help='target distribution select option')

    # system setting
    parser.add_argument('--mp',  type=str_to_bool, default=False, help='multi processing')
    parser.add_argument('--device', default='cuda', help='set specific GPU number of CPU')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--log_dir', default='log', help='Log directory')
    parser.add_argument('--wandb', type=str_to_bool, default=False, help='wandb')

    args = parser.parse_args()
    return args