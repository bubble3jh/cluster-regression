
## Coarse Search
## Cos Anneal
for lr_init in 1e-2 1e-3 1e-4
do
for wd in 1e-3 1e-4 
do
for drop_out in 0.0 0.1
do
for hidden_dim in 128
do
for num_features in 128
do
for eval_date in 0 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=4 "/mlainas/teang1995/anaconda3/envs/cluster/bin/python3" main.py --model=mlp --hidden_dim=${hidden_dim} --optim=adam --lr_init=${lr_init} --wd=${wd} --epochs=200 --scheduler=cos_anneal --t_max=200 --drop_out=${drop_out} --eval_date=${eval_date}
done
done
done
done
done
done