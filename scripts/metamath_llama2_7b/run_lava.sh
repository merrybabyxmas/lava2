#!/bin/bash

# 1. 경로 및 기본 환경 설정
PROJECT_ROOT="/home/dongwoo38/PiSSA"
cd $PROJECT_ROOT

BASE_MODEL="meta-llama/Llama-2-7b-hf"
DATA_PATH="fxmeng/pissa-dataset"
SEED=42

# 2. LAVA 하이퍼파라미터 및 DType 설정
LAMBDA_VIB=0.05
LAMBDA_STAB=0.1
RANK=128

# 모델 및 어댑터 정밀도 설정 (선택: fp32, bf16, fp16, int4, int8)
BASE_DTYPE="int4"    # 베이스 모델 dtype
ADAPTER_DTYPE="fp32" # 어댑터 모델 dtype

# 3. 출력 경로 (설정값이 한눈에 보이도록 폴더명 구성)
OUTPUT_PATH="output/metamath-LAVA-r${RANK}-B_${BASE_DTYPE}-A_${ADAPTER_DTYPE}-vib${LAMBDA_VIB}-seed${SEED}"

# 4. 환경 변수 설정
export WANDB_PROJECT=NLG-metamath-final
export WANDB_NAME="LAVA_Llama2_7B_r${RANK}_B-${BASE_DTYPE}_A-${ADAPTER_DTYPE}_vib${LAMBDA_VIB}_seed${SEED}"

# 분산 학습 및 CUDA 이슈 방지 설정
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_IGNORE_DISABLED_P2P=1
export DS_SKIP_CUDA_CHECK=1 # CUDA 버전 불일치 에러 우회

while true; do
    RANDOM_PORT=$(shuf -i 10000-60000 -n 1)
    if ! ss -ant | grep -q ":$RANDOM_PORT "; then
        export MASTER_PORT=$RANDOM_PORT
        break
    fi
done

echo "Using Master Port: $MASTER_PORT"


# 5. DeepSpeed 실행 명령어
# --include=localhost:2 를 통해 특정 GPU(2번)만 사용
# 이미 사용중이면 포트 번호 변경해서 진행
deepspeed --master_port=$MASTER_PORT --include=localhost:2 train.py \
  --deepspeed configs/ds_config_zero2_no_offload.json \
  --full_finetune False \
  --model_name_or_path $BASE_MODEL \
  --seed $SEED \
  --data_seed $SEED \
  --base_dtype $BASE_DTYPE \
  --adapter_dtype $ADAPTER_DTYPE \
  --init_weights lava \
  --lora_rank $RANK \
  --lora_alpha 128 \
  --lora_dropout 0 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --data_path $DATA_PATH \
  --sub_task metamath:100000 \
  --dataset_split train \
  --dataset_field instruction output \
  --output_dir $OUTPUT_PATH \
  --num_train_epochs 1 \
  --model_max_length 512 \
  --per_device_train_batch_size 2 \
  --gradient_checkpointing True \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 1 \
  --report_to wandb \
  --optim adamw_torch \
  --merge False \
  --lambda_vib $LAMBDA_VIB \
  --lambda_stab $LAMBDA_STAB