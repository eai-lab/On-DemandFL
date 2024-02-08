num_rounds=300
device=0
alpha=0.1
seed=42

for alpha in 0.1 0.5 1.0
do
    python3 main.py\
        --dataset_name=cifar10\
        --num_clients=100\
        --fraction=0.1\
        --mp=True\
        --seed=${seed}\
        --tm_local_bs=10\
        --tm_local_ep=1\
        --tm_lr=0.01\
        --dm_local_bs=5\
        --dm_local_ep=1\
        --dm_lr=0.001\
        --dm_optimizer=SGD\
        --dm_criterion=MSELoss\
        --device=${device}\
        --alpha=${alpha}\
        --num_rounds=${num_rounds}\
        --init_round=100\
        --wandb=True\
        --wandb_project_name=on-demand-fl
done
