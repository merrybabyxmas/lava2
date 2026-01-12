#!/bin/bash

# 1. 경로 및 기본 환경 설정
PROJECT_ROOT="/home/dongwoo38/PiSSA"
cd $PROJECT_ROOT

BASE_MODEL="meta-llama/Llama-2-7b-hf"
DATA_PATH="fxmeng/pissa-dataset"
SEED=42

# 2. LAVA 하이퍼파라미터 및 DType 설정
LAMBDA_VIB=1.0
LAMBDA_STAB=0.1
RANK=128
ALPHA=16

# r=8, alpha=4 기준 최적 Bound인 16 설정 (혹은 비워두어 자동계산 유도)

BASE_DTYPE="int4"
ADAPTER_DTYPE="fp32"

# 3. 출력 경로 및 WandB 이름에 Alpha 추가
# 이름 예시: LAVA_Llama2_7B_r128_a16_B-int4_vib0.0005
WANDB_NAME="[CONVO]LAVA_Llama2_7B_r${RANK}_a${ALPHA}_B-${BASE_DTYPE}_A-${ADAPTER_DTYPE}_vib${LAMBDA_VIB}_seed${SEED}"
OUTPUT_PATH="output/conversation-LAVA-mistral-r${RANK}-B_${BASE_DTYPE}-A_${ADAPTER_DTYPE}-vib${LAMBDA_VIB}-seed${SEED}"

# 4. 환경 변수 설정
export WANDB_PROJECT=NLG-conversation-ver3
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
# 🚀 최적화 포인트:
# - All Linear Layers 적용 (q,k,v,o,gate,up,down)
# - Total Batch Size 128로 맞춤 (1 GPU * 2 BS * 64 Acc = 128) 
# - 만약 GPU 2개를 쓴다면 --include=localhost:0,1 로 바꾸고 Acc를 32로 낮추세요.
deepspeed --master_port=$MASTER_PORT --include=localhost:3 train.py \
  --deepspeed configs/ds_config_zero2_no_offload.json \
  --full_finetune False \
  --model_name_or_path $BASE_MODEL \
  --seed $SEED \
  --data_seed $SEED \
  --base_dtype $BASE_DTYPE \
  --adapter_dtype $ADAPTER_DTYPE \
  --init_weights lava \
  --lora_rank $RANK \
  --lora_alpha $ALPHA \
  --lora_dropout 0 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --data_path $DATA_PATH \
  --sub_task conversation \
  --dataset_split train \
  --dataset_field instruction output \
  --output_dir $OUTPUT_PATH \
  --num_train_epochs 1 \
  --model_max_length 256 \
  --per_device_train_batch_size 4 \
  --gradient_checkpointing True \
  --gradient_accumulation_steps 32 \
  --learning_rate 1e-4 \
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
  --lambda_stab $LAMBDA_STAB \
  --lambda_latent_stability 1.0 