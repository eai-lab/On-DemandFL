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
    parser.add_argument('--rounds', type=int, default=100, help="number of round of training")
    parser.add_argument('--num_clients', type=int, default=100, help='number of client (K)')
    parser.add_argument('--fraction', type=float, default=0.1, help='fraction of client (C)')
    parser.add_argument('--local_ep', type=int, default=1, help='the number of local epochs of task model: E_t')
    parser.add_argument('--local_bs', type=int, default=10, help='batch size of local target model: B_t')

    # target model arguments
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate of task model')
    parser.add_argument('--criterion', default='CrossEntropyLoss', help='criterion of task model')
    parser.add_argument('--optimizer', default='SGD', help='optimizer of task model')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of optimizer of task model')
    parser.add_argument('--finetune_ep', type=int, default=1)

    # dataset setting
    parser.add_argument('--dataset_name', help='dataset name')
    parser.add_argument('--alpha',  type=float, default=1, help='alpha of dirichlet distribution')
    parser.add_argument('--target_dist_op', type=int, default=4, help='target distribution select option')

    # system setting
    parser.add_argument('--mp',  type=str_to_bool, default=False, help='multi processing')
    parser.add_argument('--device', default='cuda', help='set specific GPU number of CPU')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--log_dir', default='log', help='Log directory')

    args = parser.parse_args()
    return args