# for lr_init in 1e-2 1e-3 1e-4
# do
# for wd in 1e-2 1e-3 1e-4
# do
# for alpha in 10 1 0.1 1e-2
# do
# for optim in 'adam' # 'adam' 'adamw' 'radam'
# do
# CUDA_VISIBLE_DEVICES=2 /mlainas/bubble3jh/anaconda3/envs/cluster/bin/python3 run_causal.py \
# --model=tarnet --use_treatment --run_group=tarnet \
# --lr_init=${lr_init} --optim=${optim} --wd=${wd} --alpha=${alpha}
# done
# done
# done
# done


for seed in 1000 999 998
do
CUDA_VISIBLE_DEVICES=2 /mlainas/bubble3jh/anaconda3/envs/cluster/bin/python3 run_causal.py \
--model=tarnet --use_treatment --run_group=tarnet \
--lr_init=1e-3 --optim=adam --wd=1e-4 --alpha=1e-2 \
--seed=${seed}
done
