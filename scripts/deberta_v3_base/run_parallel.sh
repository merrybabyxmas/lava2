#!/bin/bash

ADAPTER=$1
GPUS=$2
PER_GPU=$3
TASKS=$4
SEEDS=$5
# 추가된 인자들을 변수에 할당
VIB=$6
STAB=$7
ALPHA=$8

GPU_ARR=(${GPUS//,/ })
i=0

for task in $TASKS; do
  for seed in $SEEDS; do
    gpu_index=$((i % ${#GPU_ARR[@]}))
    gpu_id=${GPU_ARR[$gpu_index]}

    echo "[RUN] GPU=$gpu_id → Task=$task, Seed=$seed, Alpha=$ALPHA, VIB=$VIB, STAB=$STAB"

    # python 실행 시 인자들을 모두 전달하도록 수정
    CUDA_VISIBLE_DEVICES=$gpu_id \
    python train_nlu.py \
        --adapter $ADAPTER \
        --task $task \
        --seed $seed \
        --lambda_vib $VIB \
        --lambda_stab $STAB \
        --alpha $ALPHA &

    i=$((i+1))
    if (( i % PER_GPU == 0 )); then
        wait
    fi
  done
done
wait