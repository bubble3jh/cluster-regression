
## Coarse Search
## Cos Anneal
for lr_init in 1e-2 1e-3 1e-4
do
for wd in 1e-3 1e-4 
do
for eval_date in 0 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=7 "/mlainas/teang1995/anaconda3/envs/cluster/bin/python3" main.py --model=linear --optim=adam --lr_init=${lr_init} --wd=${wd} --epochs=200 --scheduler=cos_anneal --t_max=200 --disable_embedding --eval_date=${eval_date}
done
done
done