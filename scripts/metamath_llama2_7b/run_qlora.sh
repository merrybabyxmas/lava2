BASE_MODEL="meta-llama/Llama-2-7b-hf"
OUTPUT_PATH="output/metamath-QLoRA-Llama-2-7B-4bit-r128-bs128"
DATA_PATH="fxmeng/pissa-dataset"
seed = 42

export WANDB_PROJECT=NLG-metamath-final
export WANDB_NAME=QLoRA_Llama2_7B_r128_seed42_bs128  # bs64 → bs128

export NCCL_DEBUG=WARN  # INFO → WARN (로그 줄이기)
export NCCL_IB_DISABLE=1
export NCCL_IGNORE_DISABLED_P2P=1  # 추가

# batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
deepspeed --master_port=16973 --include=localhost:0,1,2,3 train.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path $BASE_MODEL \
    --full_finetune False \
    --bf16 \
    --bits 4 \
    --seed $SEED \
    --data_seed $SEED \
    --init_weights Gaussian \
    --target_modules "q_proj,v_proj,k_proj,o_proj" \
    --lora_rank 128 \
    --lora_alpha 128 \
    --lora_dropout 0 \
    --data_path $DATA_PATH \
    --dataset_split "train"\
    --sub_task metamath:100000 \
    --dataset_field instruction output \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 1 \
    --model_max_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \


