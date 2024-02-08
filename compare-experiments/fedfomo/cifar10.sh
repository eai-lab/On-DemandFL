rounds=500
device=0

for alpha in 0.5
do
  nohup python3 fedfomo.py\
    --dataset_name=cifar10\
    --num_clients=100\
    --fraction=0.1\
    --mp=True\
    --local_bs=10\
    --local_ep=1\
    --device=${device}\
    --alpha=${alpha}\
    --rounds=${rounds}\
    > cifar10-A${alpha}.out 2>&1 &
done