## Best
# python3 main.py --model=GAT --hidden_dim=8 --optim=adam --save_path=./exp_result/GAT-adam/ --lr_init=0.0005 --wd=1e-2 --drop_out=0.5 --epochs=200 --scheduler=cos_anneal --t_max=100

### Coarse Search
# cosine annealing scheduler
for lr_init in 0.01 0.001 0.0001 0.00001
do
    for wd in 1e-3 1e-4 1e-5
    do
        for drop_out in 0.0 0.1 0.5 0.6
        do
        CUDA_VISIBLE_DEVICES=6 python3 main.py --model=GAT --optim=adam --save_path=./exp_result/GAT-adam/ --lr_init=${lr_init} --wd=${wd} --drop_out=${drop_out} --epochs=200 --scheduler=cos_anneal --t_max=100
        done
    done
done


## Fine-Grained Search (1)
# for lr_init in 0.005 0.001 0.0005
# do
#     for wd in 1e-2 5e-3 1e-3 5e-4
#     do
#         for drop_out in 0.4 0.5 0.6
#         do
#         python3 main.py --model=GAT --optim=adam --save_path=./exp_result/GAT-adam/ --lr_init=${lr_init} --wd=${wd} --drop_out=${drop_out} --epochs=200 --scheduler=cos_anneal --t_max=100
#         done
#     done
# done



## Fine-Grained Search (2)
# for lr_init in 0.0005
# do
#     for wd in 5e-2 1e-2 5e-3
#     do
#         for drop_out in 0.4 0.5 0.6
#         do
#         python3 main.py --model=GAT --optim=adam --save_path=./exp_result/GAT-adam/ --lr_init=${lr_init} --wd=${wd} --drop_out=${drop_out} --epochs=200 --scheduler=cos_anneal --t_max=100
#         done
#     done
# done