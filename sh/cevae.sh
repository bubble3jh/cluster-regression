#!/bin/bash

GPU_IDS=(3 4 5 6)  # 사용할 GPU ID 리스트
IDX=0
num_epochs=200
sweep_group="make_transformer-like"
for lambda1 in 1
do
  for lambda2 in 1
  do
    for lambda3 in 1
    do
    for elbo_lambda1 in 0 
    do
    for elbo_lambda2 in 0 
    do
    for elbo_lambda3 in 0 
    do
    for elbo_lambda4 in 0 1
    do
    for elbo_lambda5 in 0 1
    do
      for learning_rate in 1e-3
      do
      for feature_dim in 32 # 16 48
      do
      for num_layers in 4 # 2 3
      do
      for beta in 0.5
      do
      for eval_model in "decoder" #"encoder" # "decoder"
      do
          latent_dim=$((feature_dim / 2))
          # 현재 GPU ID 선택
          CUDA_VISIBLE_DEVICES=${GPU_IDS[$IDX]} python cevae.py \
          --learning_rate ${learning_rate} \
          --lambda1 ${lambda1} \
          --lambda2 ${lambda2} \
          --lambda3 ${lambda3} \
          --elbo_lambda1 ${elbo_lambda1} \
          --elbo_lambda2 ${elbo_lambda2} \
          --elbo_lambda3 ${elbo_lambda3} \
          --elbo_lambda4 ${elbo_lambda4} \
          --elbo_lambda5 ${elbo_lambda5} \
          --beta ${beta} \
          --sweep_group ${sweep_group} \
          --feature_dim ${feature_dim} \
          --latent_dim ${latent_dim} \
          --hidden_dim ${feature_dim} \
          --num-layers ${num_layers} \
          --eval_model ${eval_model} \
          --num_epochs ${num_epochs} &
          
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

wait 