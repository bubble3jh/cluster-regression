## Best


### Coarse Search
# Constant
for lr_init in 1e-2 1e-3 1e-4 1e-5
do
    for wd in 1e-3 1e-4 1e-5
    do
        for scaling in 'minmax' 'normalization'
        do
            for emb in '--emb' ''
            do
            CUDA_VISIBLE_DEVICES=5 python3 main.py --model=mlp --optim=adam --lr_init=${lr_init} --wd=${wd} --epochs=200 --scheduler=constant --scaling=${scaling} ${emb}
            done
        done
    done
done

# Cos Anneal
for lr_init in 1e-2 1e-3 1e-4 1e-5
do
    for wd in 1e-3 1e-4 1e-5
    do
        for scaling in 'minmax' 'normalization'
        do
            for emb in '--emb' ''
            do
            CUDA_VISIBLE_DEVICES=5 python3 main.py --model=mlp --optim=adam --lr_init=${lr_init} --wd=${wd} --epochs=200 --scheduler=constant --scaling=${scaling} ${emb}
            done
        done
    done
done