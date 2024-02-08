rounds=100
device=0

for alpha in 0.5
do
  python3 clf.py\
    --dataset_name=fashionmnist\
    --num_clients=100\
    --fraction=0.1\
    --mp=True\
    --tm_local_bs=10\
    --tm_local_ep=1\
    --device=${device}\
    --alpha=${alpha}\
    --rounds=${rounds}\
    > fashionmnist-A${alpha}.out 2>&1
done