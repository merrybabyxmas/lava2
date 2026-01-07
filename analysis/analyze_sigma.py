import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    model,
    "output/metamath-LAVA-Llama-2-7b-r128-seed42-4bit/checkpoint-2000/adapter_model"
)

print("=" * 60)
print("σ (Standard Deviation) Analysis")
print("=" * 60)

sigmas = []
for name, param in model.named_parameters():
    if 'b_logvar' in name:
        sigma = torch.exp(0.5 * param.data)
        sigmas.append(sigma)
        print(f"{name[:50]}...")
        print(f"  mean: {sigma.mean():.4f}")
        print(f"  std:  {sigma.std():.4f}")
        print(f"  min:  {sigma.min():.4f}")
        print(f"  max:  {sigma.max():.4f}")

all_sigmas = torch.cat([s.flatten() for s in sigmas])
print(f"\nOverall Statistics:")
print(f"  Mean σ: {all_sigmas.mean():.4f}")
print(f"  Std σ:  {all_sigmas.std():.4f}")

# 해석
if all_sigmas.mean() < 0.1:
    print("\n✅ σ가 작음 → Training-Inference gap 작을 것")
elif all_sigmas.mean() < 0.3:
    print("\n⚠️ σ가 중간 → 약간의 gap 예상")
else:
    print("\n🚨 σ가 큼 → Training-Inference gap 심각할 수 있음")
