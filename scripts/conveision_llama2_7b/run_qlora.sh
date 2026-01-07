BASE_MODEL="meta-llama/Llama-2-7b-hf"
OUTPUT_PATH="output/conversation-QLoRA-Llama-2-7B-4bit-r128-seed42-bs64"
DATA_PATH="fxmeng/pissa-dataset"

export CUDA_VISIBLE_DEVICES=0,1,2,3
SEED=42


export WANDB_PROJECT=NLG-conversation-ver3
export WANDB_NAME=QLoRA-Llama-2-7B-4bit-r128-seed42-bs64  # bs64 → bs128

export NCCL_DEBUG=WARN  # INFO → WARN (로그 줄이기)
export NCCL_IB_DISABLE=1
export NCCL_IGNORE_DISABLED_P2P=1  # 추가


# batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
deepspeed --master_port=16970 --include=localhost:0,1,2,3 train.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path $BASE_MODEL \
    --full_finetune False \
    --seed $SEED \
    --data_seed $SEED \
    --fp16 \
    --bits 4 \
    --init_weights gaussian \
    --target_modules "q_proj,v_proj,k_proj,o_proj" \
    --lora_rank 128 \
    --lora_alpha 128 \
    --lora_dropout 0 \
    --data_path $DATA_PATH \
    --dataset_split "train"\
    --sub_task conversation \
    --dataset_field instruction output \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 1 \
    --model_max_length 256 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
