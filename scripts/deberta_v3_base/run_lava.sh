#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

GPUS="2"
PER_GPU=1
SEEDS="19"
# TASKS="sst2 cola qnli rte stsb mnli qqp"
TASKS="mrpc"

# TASKS="mrpc cola"


LAMBDA_VIB=0.1
LAMBDA_STAB=1.0  
ALPHA=4 # <-- 여기서 직접 지정하거나, 
          # 비워두고 싶다면 train_nlu.py에서 자동 계산되도록 스크립트 수정 가능

# run_parallel.sh 호출 시 마지막에 ALPHA 추가
bash /home/dongwoo38/PiSSA/scripts/deberta_v3_base/run_parallel.sh \
     lava $GPUS $PER_GPU "$TASKS" "$SEEDS" "$LAMBDA_VIB" "$LAMBDA_STAB" "$ALPHA"



