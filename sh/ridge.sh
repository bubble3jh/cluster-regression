
# ## Coarse Search
## Cos Anneal
for lr_init in 1e-2 1e-3 1e-4
do
for wd in 1e-3 1e-4 
do
for lamb in 10
do
for num_features in 128
do
CUDA_VISIBLE_DEVICES=3 "/mlainas/teang1995/anaconda3/envs/cluster/bin/python3" main.py --model=ridge --optim=adam --lr_init=${lr_init} --wd=${wd} --epochs=200 --scheduler=cos_anneal --t_max=200  --lamb=${lamb} --num_features=${num_features}
done
done
done
done