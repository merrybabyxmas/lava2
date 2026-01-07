#!/usr/bin/env python3
"""
LAVA 초기화 테스트
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model
from peft.tuners.lava.config import LavaConfig
from peft import TaskType

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
    device_map={"": 0}
)

print("\nInitializing LAVA...")
peft_config = LavaConfig(
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, peft_config)

print("\n" + "="*60)
print("BEFORE manual fix:")
print("="*60)

lava_params = [(n, p.requires_grad, p.device, p.dtype) 
               for n, p in model.named_parameters() 
               if 'lava' in n.lower() and p.dtype.is_floating_point]

print(f"Total LAVA float params: {len(lava_params)}")
trainable_before = sum(1 for _, req_grad, _, _ in lava_params if req_grad)
print(f"Trainable: {trainable_before}")

print("\nFirst 5 params:")
for name, req_grad, device, dtype in lava_params[:5]:
    print(f"  {name}")
    print(f"    requires_grad={req_grad}, device={device}, dtype={dtype}")

# 수동 fix 적용
print("\n" + "="*60)
print("Applying manual fix...")
print("="*60)

trainable_params = []
for name, param in model.named_parameters():
    if 'lava' in name.lower():
        if param.dtype.is_floating_point:
            param.requires_grad = True
            trainable_params.append(name)
            
            if param.device.type == 'cpu':
                param.data = param.data.to('cuda:0')
        else:
            param.requires_grad = False
    else:
        param.requires_grad = False

print(f"\nSet {len(trainable_params)} LAVA parameters to trainable")

print("\n" + "="*60)
print("AFTER manual fix:")
print("="*60)

lava_params_after = [(n, p.requires_grad, p.device, p.dtype) 
                     for n, p in model.named_parameters() 
                     if 'lava' in n.lower() and p.dtype.is_floating_point]

trainable_after = sum(1 for _, req_grad, _, _ in lava_params_after if req_grad)
print(f"Trainable: {trainable_after}")

print("\nFirst 5 params:")
for name, req_grad, device, dtype in lava_params_after[:5]:
    print(f"  {name}")
    print(f"    requires_grad={req_grad}, device={device}, dtype={dtype}")

# 총 학습 가능 파라미터 수
total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n✅ Total trainable parameters: {total_trainable:,}")

if trainable_after == 0:
    print("\n❌ FAILED: No trainable parameters!")
elif trainable_after > 0:
    print(f"\n✅ SUCCESS: {trainable_after} trainable parameters!")
