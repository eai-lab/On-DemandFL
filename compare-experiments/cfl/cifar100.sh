rounds=1500
device=0

for alpha in 0.1
do
  nohup python3 clf.py\
    --dataset_name=cifar100\
    --num_clients=100\
    --fraction=0.1\
    --mp=True\
    --tm_local_bs=10\
    --tm_local_ep=1\
    --device=${device}\
    --alpha=${alpha}\
    --rounds=${rounds}\
    > cifar100-A${alpha}.out 2>&1 &
done