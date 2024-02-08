rounds=500
device=0
alpha=0.1

for alpha in 0.1 0.5 1
do
  python3 fedmtl.py\
    --dataset_name=fashionmnist\
    --num_clients=100\
    --fraction=0.1\
    --mp=True\
    --local_bs=10\
    --local_ep=1\
    --device=${device}\
    --alpha=${alpha}\
    --rounds=${rounds}\
    > fashionmnist-A${alpha}.out 2>&1
done