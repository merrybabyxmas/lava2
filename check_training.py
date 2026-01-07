#!/usr/bin/env python3
"""
학습 상태 확인
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. 모델 로드
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map={"": 0}
)
adapter_path = "output/conversation-LAVA-Llama-2-7b-r128/checkpoint-2000/adapter_model"
model = PeftModel.from_pretrained(model, adapter_path)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# 2. LAVA 파라미터 확인
print("\n" + "="*60)
print("LAVA Parameters:")
print("="*60)

trainable_count = 0
for name, param in model.named_parameters():
    if 'lava' in name.lower():
        print(f"{name}: shape={param.shape}, requires_grad={param.requires_grad}")
        if param.requires_grad:
            trainable_count += param.numel()

print(f"\nTotal trainable LAVA params: {trainable_count:,}")

# 3. 간단한 테스트
print("\n" + "="*60)
print("Testing Generation:")
print("="*60)

PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

test_q = "What is the capital of France?"
prompt = PROMPT.format(instruction=test_q)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

print(f"\nQuestion: {test_q}")
print("Generating...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"\nFull output:\n{full_text}")

if "### Response:" in full_text:
    answer = full_text.split("### Response:")[-1].strip()
    print(f"\nExtracted answer:\n{answer}")
else:
    print("\n⚠️  No '### Response:' found in output!")

# 4. Loss 확인 (가능하면)
if hasattr(model, 'base_model'):
    if hasattr(model.base_model, 'model'):
        base = model.base_model.model
        print(f"\nBase model config:")
        print(f"  vocab_size: {base.config.vocab_size}")
        print(f"  hidden_size: {base.config.hidden_size}")
