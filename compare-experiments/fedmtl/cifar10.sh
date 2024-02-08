rounds=500
device=0
alpha=0.1

for alpha in 0.1 0.5 1
do
  python3 fedmtl.py\
    --dataset_name=cifar10\
    --num_clients=100\
    --fraction=0.1\
    --mp=True\
    --local_bs=10\
    --device=${device}\
    --alpha=${alpha}\
    --rounds=${rounds}\
    > cifar10-A${alpha}.out 2>&1
done