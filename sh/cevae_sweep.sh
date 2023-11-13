#!/bin/bash
GPU_IDS=(6 7)  # 사용할 GPU ID 리스트
IDX=0

for pred_layers in 1 
do
for shared_layers in 1 2 3
do
  for num_layers in 1 2 3
  do
    for embedding_dim in 16 32 64
    do
      for latent_dim in 16 32 64
        do
          for hidden_dim in 16 32 64
          do
          # 현재 GPU ID 선택
          CUDA_VISIBLE_DEVICES=${GPU_IDS[$IDX]} /mlainas/bubble3jh/anaconda3/envs/cluster/bin/python cevae_scratch.py \
          --pred_layers ${pred_layers} \
          --shared_layers ${shared_layers} \
          --num_layers ${num_layers} \
          --embedding_dim ${embedding_dim} \
          --latent_dim ${latent_dim} \
          --hidden_dim ${hidden_dim} &
          
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

wait  # 마지막 배치의 모든 작업이 완료될 때까지 기다림