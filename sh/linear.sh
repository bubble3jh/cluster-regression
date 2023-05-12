
## Coarse Search
## Cos Anneal
for lr_init in 1e-2 1e-3 1e-4
do
for wd in 1e-3 1e-4 1e-5
do
for hidden_dim in 16 32 64
do
for num_features in 128
do
CUDA_VISIBLE_DEVICES=7 "/mlainas/teang1995/anaconda3/envs/cluster/bin/python3" main.py --model=linear --optim=adam --lr_init=${lr_init} --wd=${wd} --epochs=200 --scheduler=cos_anneal --t_max=200 
done
done
done
done