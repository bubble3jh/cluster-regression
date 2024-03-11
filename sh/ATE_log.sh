run_group="CE_log"
seed=1000
python main.py --model=mlp --hidden_dim=256 --optim=adam --lr_init=1e-2 --wd=1e-3 --epochs=200 --scheduler=cos_anneal --t_max=200 --drop_out=0.0 --num_layers=3 --run_group=${run_group} --seed=${seed}
python main.py --model=ridge --optim=adam --lr_init=1e-2 --wd=1e-5 --epochs=200 --scheduler=cos_anneal --t_max=200 --lamb=1 --hidden_dim=64 --num_features=128 --run_group=${run_group} --seed=${seed}
python main.py --model=linear --optim=adam --lr_init=1e-2 --wd=1e-3 --epochs=200 --scheduler=cos_anneal --t_max=200 --hidden_dim=64 --num_features=128 --run_group=${run_group} --seed=${seed}
python main.py --model=cet --hidden_dim=256 --optim=adam --lr_init=1e-2 --wd=5e-3 --epochs=300 --scheduler=cos_anneal --t_max=300 --drop_out=0.0 --num_layers=1 --num_features=64 --num_heads=4 --lambdas 1 1e-3 1e-6 --use_treatment --MC_sample=1 --run_group=${run_group} --seed=${seed}
python main.py --model=transformer --hidden_dim=128 --optim=adam --lr_init=1e-3 --wd=1e-2 --epochs=200 --scheduler=cos_anneal --t_max=200 --drop_out=0.0 --num_layers=2 --num_heads=4 --run_group=${run_group} --seed=${seed}

seed=999
python main.py --model=mlp --hidden_dim=256 --optim=adam --lr_init=1e-2 --wd=1e-3 --epochs=200 --scheduler=cos_anneal --t_max=200 --drop_out=0.0 --num_layers=3 --run_group=${run_group} --seed=${seed}
python main.py --model=ridge --optim=adam --lr_init=1e-2 --wd=1e-5 --epochs=200 --scheduler=cos_anneal --t_max=200 --lamb=1 --hidden_dim=64 --num_features=128 --run_group=${run_group} --seed=${seed}
python main.py --model=linear --optim=adam --lr_init=1e-2 --wd=1e-3 --epochs=200 --scheduler=cos_anneal --t_max=200 --hidden_dim=64 --num_features=128 --run_group=${run_group} --seed=${seed}
python main.py --model=cet --hidden_dim=256 --optim=adam --lr_init=1e-2 --wd=5e-3 --epochs=300 --scheduler=cos_anneal --t_max=300 --drop_out=0.0 --num_layers=1 --num_features=64 --num_heads=4 --lambdas 1 1e-3 1e-6 --use_treatment --MC_sample=1 --run_group=${run_group} --seed=${seed}
python main.py --model=transformer --hidden_dim=128 --optim=adam --lr_init=1e-3 --wd=1e-2 --epochs=200 --scheduler=cos_anneal --t_max=200 --drop_out=0.0 --num_layers=2 --num_heads=4 --run_group=${run_group} --seed=${seed}

seed=998
python main.py --model=mlp --hidden_dim=256 --optim=adam --lr_init=1e-2 --wd=1e-3 --epochs=200 --scheduler=cos_anneal --t_max=200 --drop_out=0.0 --num_layers=3 --run_group=${run_group} --seed=${seed}
python main.py --model=ridge --optim=adam --lr_init=1e-2 --wd=1e-5 --epochs=200 --scheduler=cos_anneal --t_max=200 --lamb=1 --hidden_dim=64 --num_features=128 --run_group=${run_group} --seed=${seed}
python main.py --model=linear --optim=adam --lr_init=1e-2 --wd=1e-3 --epochs=200 --scheduler=cos_anneal --t_max=200 --hidden_dim=64 --num_features=128 --run_group=${run_group} --seed=${seed}
python main.py --model=cet --hidden_dim=256 --optim=adam --lr_init=1e-2 --wd=5e-3 --epochs=300 --scheduler=cos_anneal --t_max=300 --drop_out=0.0 --num_layers=1 --num_features=64 --num_heads=4 --lambdas 1 1e-3 1e-6 --use_treatment --MC_sample=1 --run_group=${run_group} --seed=${seed}
python main.py --model=transformer --hidden_dim=128 --optim=adam --lr_init=1e-3 --wd=1e-2 --epochs=200 --scheduler=cos_anneal --t_max=200 --drop_out=0.0 --num_layers=2 --num_heads=4 --run_group=${run_group} --seed=${seed}

# seed=900
# python main.py --model=mlp --hidden_dim=256 --optim=adam --lr_init=1e-2 --wd=1e-3 --epochs=200 --scheduler=cos_anneal --t_max=200 --drop_out=0.0 --num_layers=3 --run_group=${run_group} --seed=${seed}
# python main.py --model=ridge --optim=adam --lr_init=1e-2 --wd=1e-5 --epochs=200 --scheduler=cos_anneal --t_max=200 --lamb=1 --hidden_dim=64 --num_features=128 --run_group=${run_group} --seed=${seed}
# python main.py --model=linear --optim=adam --lr_init=1e-2 --wd=1e-3 --epochs=200 --scheduler=cos_anneal --t_max=200 --hidden_dim=64 --num_features=128 --run_group=${run_group} --seed=${seed}
# python main.py --model=cet --hidden_dim=256 --optim=adam --lr_init=1e-2 --wd=1e-3 --epochs=300 --scheduler=cos_anneal --t_max=300 --drop_out=0.0 --num_layers=1 --num_features=64 --num_heads=4 --lambdas 1 1 0 --use_treatment --MC_sample=1 --run_group=${run_group} --seed=${seed}
# python main.py --model=transformer --hidden_dim=128 --optim=adam --lr_init=1e-3 --wd=1e-2 --epochs=200 --scheduler=cos_anneal --t_max=200 --drop_out=0.0 --num_layers=2 --num_heads=4 --run_group=${run_group} --seed=${seed}

# seed=800
# python main.py --model=mlp --hidden_dim=256 --optim=adam --lr_init=1e-2 --wd=1e-3 --epochs=200 --scheduler=cos_anneal --t_max=200 --drop_out=0.0 --num_layers=3 --run_group=${run_group} --seed=${seed}
# python main.py --model=ridge --optim=adam --lr_init=1e-2 --wd=1e-5 --epochs=200 --scheduler=cos_anneal --t_max=200 --lamb=1 --hidden_dim=64 --num_features=128 --run_group=${run_group} --seed=${seed}
# python main.py --model=linear --optim=adam --lr_init=1e-2 --wd=1e-3 --epochs=200 --scheduler=cos_anneal --t_max=200 --hidden_dim=64 --num_features=128 --run_group=${run_group} --seed=${seed}
# python main.py --model=cet --hidden_dim=256 --optim=adam --lr_init=1e-2 --wd=1e-3 --epochs=300 --scheduler=cos_anneal --t_max=300 --drop_out=0.0 --num_layers=1 --num_features=64 --num_heads=4 --lambdas 1 1 0 --use_treatment --MC_sample=1 --run_group=${run_group} --seed=${seed}
# python main.py --model=transformer --hidden_dim=128 --optim=adam --lr_init=1e-3 --wd=1e-2 --epochs=200 --scheduler=cos_anneal --t_max=200 --drop_out=0.0 --num_layers=2 --num_heads=4 --run_group=${run_group} --seed=${seed}
