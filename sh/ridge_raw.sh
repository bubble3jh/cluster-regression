
# ## Coarse Search
## Cos Anneal
for lr_init in 1e-2 1e-3 1e-4
do
for wd in 1e-3 1e-4 
do
for lamb in 0.1 1 10
do
for eval_date in 0 1 2 3 4 5
do
python3 main.py --model=ridge --optim=adam --lr_init=${lr_init} --wd=${wd} --epochs=200 --scheduler=cos_anneal --t_max=200  --lamb=${lamb} --disable_embedding --eval_date=${eval_date}
done
done
done
done