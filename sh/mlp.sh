
## Coarse Search
## Cos Anneal
for lr_init in 1e-3 1e-4 1e-2
do
for wd in 1e-3 1e-4 
do
for drop_out in 0.0 #0.1 #0.5
do
for hidden_dim in 256 #512
do
for num_features in 128
do
for num_layers in 3
do
python3 main.py --model=mlp --hidden_dim=${hidden_dim} --optim=adam --lr_init=${lr_init} --wd=${wd} --epochs=200 --scheduler=cos_anneal --t_max=200 --drop_out=${drop_out} --num_layers=${num_layers}
done
done
done
done
done
done