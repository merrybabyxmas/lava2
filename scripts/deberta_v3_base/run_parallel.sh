#!/bin/bash

ADAPTER=$1     # lava, lora, pissa 등
GPUS=$2
PER_GPU=$3
TASKS=$4
SEEDS=$5

GPU_ARR=(${GPUS//,/ })

i=0

for task in $TASKS; do
  for seed in $SEEDS; do

    gpu_index=$((i % ${#GPU_ARR[@]}))
    gpu_id=${GPU_ARR[$gpu_index]}

    echo "[RUN] GPU=$gpu_id → python train_nlu.py --adapter $ADAPTER --task $task --seed $seed"

    CUDA_VISIBLE_DEVICES=$gpu_id \
    python train_nlu.py \
        --adapter $ADAPTER \
        --task $task \
        --seed $seed &

    i=$((i+1))

    # GPU 당 PER_GPU 개수만큼 실행되면 대기
    if (( i % PER_GPU == 0 )); then
        wait
    fi

  done
done

wait
