import torch
from transformers import AutoModelForCausalLM
from peft import get_peft_model
from peft.tuners.lava.config import LavaConfig
from peft import TaskType

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
    device_map="auto"
)

peft_config = LavaConfig(
    r=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, peft_config)

# 수동으로 requires_grad 설정 (train.py 로직)
for name, param in model.named_parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if 'lava' in name.lower() and param.dtype.is_floating_point:
        param.requires_grad = True

# 확인
print("=" * 60)
print("Trainable Parameters")
print("=" * 60)

trainable = []
for name, param in model.named_parameters():
    if param.requires_grad:
        trainable.append(name)

print(f"Total trainable: {len(trainable)}")
print(f"\nFirst 20:")
for name in trainable[:20]:
    print(f"  - {name}")

# b_logvar 확인
b_logvar_trainable = [n for n in trainable if 'b_logvar' in n]
w_mu_trainable = [n for n in trainable if 'W_mu.weight' in n]
w_o_trainable = [n for n in trainable if 'W_o.weight' in n]

print(f"\nParameter type counts:")
print(f"  b_logvar: {len(b_logvar_trainable)}")
print(f"  W_mu.weight: {len(w_mu_trainable)}")
print(f"  W_o.weight: {len(w_o_trainable)}")

if len(w_mu_trainable) == 0:
    print("\n🚨 ERROR: W_mu.weight is NOT trainable!")
    print("   → The current train.py logic is BROKEN")
else:
    print("\n✅ SUCCESS: All LAVA parameters are trainable!")
