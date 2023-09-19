#!/bin/bash

GPU_IDS=(0 1 2 3 4 5 6)  # 사용할 GPU ID 리스트
IDX=0
num_epochs=200
for lambda1 in 0.1 1 10
do
  for lambda2 in 0.1 1 10
  do
    for lambda3 in 0.1 1 10
    do
      for learning_rate in 1e-2 1e-3 1e-4
      do
      for beta in -0.5 0 0.5 1.5
      do
          # 현재 GPU ID 선택
          CUDA_VISIBLE_DEVICES=${GPU_IDS[$IDX]} python cevae.py \
          --learning_rate ${learning_rate} \
          --lambda1 ${lambda1} \
          --lambda2 ${lambda2} \
          --lambda3 ${lambda3} \
          --beta ${beta} \
          --num_epochs ${num_epochs} &
          --beta ${beta} \
          --num_epochs ${num_epochs}
          
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

wait 