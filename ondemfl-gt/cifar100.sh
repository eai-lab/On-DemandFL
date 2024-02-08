init_round=1200
alpha=1.0
dm_pred_method=avg_dist_pred 
device=0
num_rounds=300
client_pred_round=200

# for op in 30 40 50 
for op in {101..210}
do
    for alpha in 0.1 0.5 1.0
    do
    python3 main.py\
        --dataset_name=cifar100\
        --init_round=${init_round}\
        --num_clients=100\
        --fraction=0.1\
        --mp=True\
        --tm_local_bs=10\
        --tm_local_ep=1\
        --dm_local_bs=5\
        --dm_local_ep=1\
        --dm_criterion=MSELoss\
        --device=${device}\
        --alpha=${alpha}\
        --num_rounds=${num_rounds}\
        --target_dist_op=$(($op-1))\
        --subset_size=10\
        --method=2\
        --dm_model_idx=0\
        --dm_pred_method=${dm_pred_method}\
        --wandb=True
    done
done