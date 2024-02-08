init_round=100
alpha=1.0
dm_pred_method=avg_dist_pred 
device=0
num_rounds=100
client_pred_round=50

# for op in 3 4 5
for op in {11..110}
do
    for alpha in 0.1 0.5 1.0
    do
    python3 main.py\
        --dataset_name=fashionmnist\
        --init_round=${init_round}\
        --client_pred_round=${client_pred_round}\
        --num_clients=100\
        --fraction=0.1\
        --mp=True\
        --tm_local_bs=10\
        --tm_local_ep=1\
        --dm_local_bs=10\
        --dm_local_ep=10\
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