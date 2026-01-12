#!/bin/bash

# 1. 경로 및 기본 환경 설정
PROJECT_ROOT="/home/dongwoo38/PiSSA"
cd $PROJECT_ROOT

BASE_MODEL="meta-llama/Llama-2-7b-hf"
DATA_PATH="fxmeng/pissa-dataset"
SEED=42

# 2. LoRA 하이퍼파라미터 및 DType 설정
# 비교를 위해 LAVA와 동일한 RANK를 사용합니다.
RANK=128
ALPHA=128  # 일반 LoRA/PiSSA는 alpha=r 또는 alpha=2r을 주로 사용합니다.
INIT_TYPE="gaussian" # "pissa" 또는 "gaussian" (일반 LoRA) 중 선택

BASE_DTYPE="int4"
ADAPTER_DTYPE="fp32"

# 3. 출력 경로 및 WandB 이름 설정 (VIB 관련 지표 제거)
WANDB_NAME="[BASELINE]${INIT_TYPE^^}_Llama2_7B_r${RANK}_a${ALPHA}_B-${BASE_DTYPE}_seed${SEED}"
OUTPUT_PATH="output/metamath-${INIT_TYPE}-Llama2-r${RANK}-a${ALPHA}-seed${SEED}"

# 4. 환경 변수 설정
export WANDB_PROJECT=NLG-comparison-baselines
export WANDB_NAME=$WANDB_NAME

# 분산 학습 및 CUDA 이슈 방지 설정
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_IGNORE_DISABLED_P2P=1
export DS_SKIP_CUDA_CHECK=1 

# 자동 포트 할당 로직
while true; do
    RANDOM_PORT=$(shuf -i 10000-60000 -n 1)
    if ! ss -ant | grep -q ":$RANDOM_PORT "; then
        export MASTER_PORT=$RANDOM_PORT
        break
    fi
done

echo "Using Master Port: $MASTER_PORT"

# 5. DeepSpeed 실행 명령어
# 🚀 수정 포인트: 
# - init_weights를 문자열 "pissa" 또는 "gaussian"으로 명시 (True 에러 방지)
# - lambda_vib, lambda_stab 등 LAVA 전용 인자 전면 제거
# - GPU 2개를 사용한다면 --include=localhost:0,1 로 설정하세요.
deepspeed --master_port=$MASTER_PORT --include=localhost:0,1 train.py \
  --deepspeed configs/ds_config_zero2_no_offload.json \
  --full_finetune False \
  --model_name_or_path $BASE_MODEL \
  --seed $SEED \
  --data_seed $SEED \
  --base_dtype $BASE_DTYPE \
  --adapter_dtype $ADAPTER_DTYPE \
  --init_weights $INIT_TYPE \
  --lora_rank $RANK \
  --lora_alpha $ALPHA \
  --lora_dropout 0 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --data_path $DATA_PATH \
  --sub_task metamath:100000 \
  --dataset_split train \
  --dataset_field instruction output \
  --output_dir $OUTPUT_PATH \
  --num_train_epochs 1 \
  --model_max_length 512 \
  --per_device_train_batch_size 4 \
  --gradient_checkpointing True \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 1 \
  --report_to wandb \
  --optim adamw_torch \
  --merge False