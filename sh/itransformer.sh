for lr_init in 1e-2 1e-3 1e-4
do
for wd in 1e-2 1e-3 1e-4
do
for hidden_dim in 128 256 512 1024
do
for num_layers in 1 2 3
do
for num_heads in 2 4 8
do
for drop_out in 0.0 0.2 0.4
do
CUDA_VISIBLE_DEVICES=7 /mlainas/bubble3jh/anaconda3/envs/cluster/bin/python3 run_itransformer.py \
--model=iTransformer --use_treatment --run_group=itransformer \
--hidden_dim=${hidden_dim} --num_layers=${num_layers} --num_heads=${num_heads} \
--lr_init=${lr_init} --optim=adam --wd=${wd} --drop_out=${drop_out}
done
done
done
done
done
done