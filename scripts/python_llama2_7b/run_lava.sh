#!/bin/bash

cd /home/dongwoo38/PiSSA

BASE_MODEL="meta-llama/Llama-2-7b-hf"
OUTPUT_PATH="output/python-LAVA-Llama-2-7b-r128-seed42-4bit-bf16"
DATA_PATH="fxmeng/pissa-dataset"
SEED=42

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3

export WANDB_PROJECT=NLG-python-final
export WANDB_NAME=LAVA_Llama2_7B_r128_seed42_4bit_bf16  # bs64 → bs128

export NCCL_DEBUG=WARN  # INFO → WARN (로그 줄이기)
export NCCL_IB_DISABLE=1
export NCCL_IGNORE_DISABLED_P2P=1  # 추가

deepspeed --master_port=16973 --include=localhost:0,1,2,3 train.py \
  --deepspeed configs/ds_config_zero2_no_offload.json \
  --full_finetune False \
  --model_name_or_path $BASE_MODEL \
  --seed $SEED \
  --data_seed $SEED \
  --bf16 \
  --init_weights lava \
  --lora_rank 128 \
  --bits 4 \
  --lora_alpha 128 \
  --lora_dropout 0 \
  --target_modules q_proj,k_proj,v_proj,o_proj \
  --data_path $DATA_PATH \
  --sub_task python \
  --dataset_split train \
  --dataset_field instruction output \
  --output_dir $OUTPUT_PATH \
  --num_train_epochs 3 \
  --model_max_length 512 \
  --per_device_train_batch_size 2 \
  --gradient_checkpointing False \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 1 \
  --report_to wandb \
  --merge False