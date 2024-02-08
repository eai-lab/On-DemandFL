rounds=1500
device=0

for alpha in 0.1 0.5 1
do
  nohup python3 fedmtl.py\
    --dataset_name=cifar100\
    --num_clients=100\
    --fraction=0.1\
    --mp=True\
    --local_bs=50\
    --device=${device}\
    --alpha=${alpha}\
    --rounds=${rounds}\
    > cifar100-A${alpha}.out 2>&1 &
done