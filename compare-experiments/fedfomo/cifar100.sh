rounds=1500
device=1

for alpha in 0.1 0.5 1.0
do
  nohup python3 fedfomo.py\
    --dataset_name=cifar100\
    --num_clients=100\
    --fraction=0.1\
    --mp=True\
    --local_bs=10\
    --local_ep=1\
    --device=${device}\
    --alpha=${alpha}\
    --rounds=${rounds}\
    > cifar100-A${alpha}.out 2>&1 &
done