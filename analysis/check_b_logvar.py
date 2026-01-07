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
    "/home/dongwoo38/PiSSA/output/metamath-LAVA-Llama-2-7b-r128-seed42-4bit/checkpoint-2000/adapter_model"
)

print("=" * 60)
print("b_logvar Analysis (raw values)")
print("=" * 60)

for name, param in model.named_parameters():
    if 'b_logvar' in name:
        data = param.data
        print(f"\n{name[:60]}...")
        print(f"  Shape: {data.shape}")
        print(f"  First 10 values: {data[:10].tolist()}")
        print(f"  Mean: {data.mean():.6f}")
        print(f"  Std:  {data.std():.6f}")
        print(f"  Min:  {data.min():.6f}")
        print(f"  Max:  {data.max():.6f}")
        print(f"  Unique values: {len(torch.unique(data))}")
        
        # σ 계산
        sigma = torch.exp(0.5 * data)
        print(f"  σ Mean: {sigma.mean():.6f}")
        print(f"  σ Std:  {sigma.std():.6f}")
        
        # 학습 여부 확인
        if data.std() < 1e-6:
            print("  ⚠️ WARNING: b_logvar가 학습되지 않았습니다!")
        
        break  # 첫 번째만 자세히 보기
