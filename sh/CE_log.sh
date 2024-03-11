run_group="CE_log"
# seed=1000

for seed in 1000 999 998
do

# CUDA_VISIBLE_DEVICES=3 python main.py --model=mlp --hidden_dim=256 --optim=adam --lr_init=1e-2 --wd=1e-3 --epochs=200 --scheduler=cos_anneal --t_max=200 --drop_out=0.0 --num_layers=3 --run_group=${run_group} --seed=${seed} &
# CUDA_VISIBLE_DEVICES=4 python main.py --model=ridge --optim=adam --lr_init=1e-2 --wd=1e-5 --epochs=200 --scheduler=cos_anneal --t_max=200 --lamb=1 --hidden_dim=64 --num_features=128 --run_group=${run_group} --seed=${seed} & 
# CUDA_VISIBLE_DEVICES=5 python main.py --model=linear --optim=adam --lr_init=1e-2 --wd=1e-3 --epochs=200 --scheduler=cos_anneal --t_max=200 --hidden_dim=64 --num_features=128 --run_group=${run_group} --seed=${seed} &
# CUDA_VISIBLE_DEVICES=7 python main.py --model=transformer --hidden_dim=128 --optim=adam --lr_init=1e-3 --wd=1e-2 --epochs=200 --scheduler=cos_anneal --t_max=200 --drop_out=0.0 --num_layers=2 --num_heads=4 --run_group=${run_group} --seed=${seed} &

# CUDA_VISIBLE_DEVICES=6 python main.py --model=cet --hidden_dim=256 --optim=adam --lr_init=1e-2 --wd=5e-3 --epochs=300 --scheduler=cos_anneal --t_max=300 --drop_out=0.0 --num_layers=1 --num_features=64 --num_heads=4 --lambdas 1 1e-3 1e-6 --use_treatment --MC_sample=1 --run_group=${run_group} --seed=${seed} &
CUDA_VISIBLE_DEVICES=6 python main.py --model=cet --filter_out_clip --hidden_dim=256 --optim=adam --lr_init=1e-2 --wd=5e-3 --epochs=300 --scheduler=cos_anneal --t_max=300 --drop_out=0.0 --num_layers=1 --num_features=64 --num_heads=4 --lambdas 1 1e-3 1e-6 --use_treatment --MC_sample=1 --run_group=${run_group} --seed=${seed} &

wait

done