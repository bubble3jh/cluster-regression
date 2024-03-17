GPU_IDS=(2 3 4 5 6 7)  # 사용할 GPU ID 리스트
IDX=0
## Coarse Search
## Cos Anneal
run_group="residual_cet"
for lr_init in 5e-2 1e-2 1e-3 #5e-3 
do
for wd in 5e-2 1e-2 5e-3 #1e-4 
do
for drop_out in 0.0 # 0.2 # 0.5
do
for hidden_dim in 128 256
do
for num_features in 64 128 256 #128 256 512
do
for num_layers in 1 2 #3 #4 5
do
for num_heads in 4 8
do
for optim in "adam"
do
for lambda1 in 1 #0.5 1 2 #0.5 1 2 
do
for lambda2 in 1e-6 #1e-6 #1 2 #0.5 1 2 
do
for lambda3 in 1e-6 #1e-6 #10
do
for unidir in "" #"" "--unidir"
do
for use_treatment in "--use_treatment" #""
do
for residual_t in "--residual_t" #""
do
CUDA_VISIBLE_DEVICES=${GPU_IDS[$IDX]} python main.py --model=cet \
--hidden_dim=${hidden_dim} \
--optim=${optim} \
--lr_init=${lr_init} \
--wd=${wd} \
--epochs=300 \
--scheduler=cos_anneal \
--t_max=300 \
--drop_out=${drop_out} \
--num_layers=${num_layers} \
--num_features=${num_features} \
--num_heads=${num_heads} \
--lambdas $lambda1 $lambda2 $lambda3 \
--run_group=${run_group} \
${unidir} \
${use_treatment} \
${residual_t} \
--MC_sample=1 &

# GPU ID를 다음 것으로 변경
IDX=$(( ($IDX + 1) % ${#GPU_IDS[@]} ))

# 모든 GPU가 사용 중이면 기다림
if [ $IDX -eq 0 ]; then
wait
fi

done
done
done
done
done
done
done
done
done
done
done
done
done
done
wait