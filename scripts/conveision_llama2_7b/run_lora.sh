BASE_MODEL="meta-llama/Llama-2-7b-hf"
OUTPUT_PATH="output/conversation-LoRA-Llama-2-7b-r16"
DATA_PATH="fxmeng/pissa-dataset"

export WANDB_PROJECT=NLG-conversation
export WANDB_NAME=LoRA_G_Llama2_7B_r16_bs64

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1

# batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
deepspeed --master_port=16971 --include=localhost:0,1,2,3 train.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path $BASE_MODEL \
    --full_finetune False \
    --init_weights gaussian \
    --bf16 \
    --target_modules "q_proj,v_proj,k_proj,o_proj" \
    --lora_rank 16 \
    --bits 8 \
    --lora_alpha 128 \
    --lora_dropout 0 \
    --data_path $DATA_PATH \
    --sub_task conversation \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 1 \
    --model_max_length 256 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing False \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --merge True \

