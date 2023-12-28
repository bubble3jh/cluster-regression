#!/bin/bash

GPU_IDS=(0 6 7)  # 사용할 GPU ID 리스트
IDX=0
num_epochs=200
sweep_group="cetransformer_complicate_new"
for lambda1 in 0.1 #1 10
do
  for lambda2 in 0.1 #1 10
  do
    for lambda3 in 1
    do
      for learning_rate in 0.001 #1e-3 5e-3 1e-2
      do
      for pred_layers in 3 4 5
      do
      for transformer_num_layers in 4 6 8
      do
      for embedding_dim in 128 256 512
      do
      for beta in 0.5
      do
      for weight_decay in 0.001 #0.01
      do
      for drop_out in 0 #0.3
      do
          # 현재 GPU ID 선택
          CUDA_VISIBLE_DEVICES=${GPU_IDS[$IDX]} python cet.py \
          --learning_rate ${learning_rate} \
          --lambdas ${lambda1} ${lambda2} ${lambda3} \
          --beta ${beta} \
          --sweep_group ${sweep_group} \
          --pred_layers ${pred_layers} \
          --transformer_num_layers ${transformer_num_layers} \
          --embedding_dim ${embedding_dim} \
          --latent_dim $((embedding_dim / 2)) \
          --num_epochs ${num_epochs} \
          --weight_decay ${weight_decay} \
          --drop_out ${drop_out} &
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


wait 