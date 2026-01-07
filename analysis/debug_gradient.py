import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

model = PeftModel.from_pretrained(
    model, 
   "/home/dongwoo38/PiSSA/output/metamath-LAVA-Llama-2-7b-r128-seed42-4bit/checkpoint-2000/adapter_model"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 간단한 forward-backward
model.train()

inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss

loss.backward()

# b_logvar의 gradient 확인
print("=" * 60)
print("Checking b_logvar gradients")
print("=" * 60)

for name, param in model.named_parameters():
    if 'b_logvar' in name:
        print(f"\n{name[:60]}...")
        print(f"  requires_grad: {param.requires_grad}")
        print(f"  grad is None: {param.grad is None}")
        
        if param.grad is not None:
            print(f"  grad mean: {param.grad.mean():.8f}")
            print(f"  grad std:  {param.grad.std():.8f}")
            print(f"  grad min:  {param.grad.min():.8f}")
            print(f"  grad max:  {param.grad.max():.8f}")
            
            if param.grad.abs().max() < 1e-10:
                print("  🚨 Gradient is effectively ZERO!")
        else:
            print("  🚨 ERROR: No gradient computed!")
        
        break  # 첫 번째만 확인
