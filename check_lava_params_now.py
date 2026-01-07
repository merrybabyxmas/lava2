import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model
from peft.tuners.lava.config import LavaConfig
from peft import TaskType

print("=" * 60)
print("Checking LAVA Parameter Setup")
print("=" * 60)

# Model loading
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# LAVA config
peft_config = LavaConfig(
    r=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, peft_config)

# Set trainable
trainable_names = []
for name, param in model.named_parameters():
    if 'lava' in name.lower():
        if param.dtype.is_floating_point:
            param.requires_grad = True
            trainable_names.append(name)
        else:
            param.requires_grad = False
    else:
        param.requires_grad = False

print(f"\nSet {len(trainable_names)} LAVA parameters to trainable")
print(f"\nFirst 10 trainable params:")
for name in trainable_names[:10]:
    print(f"  - {name}")

# Count by type
b_mu_count = sum(1 for n in trainable_names if 'b_mu' in n and 'b_logvar' not in n)
b_logvar_count = sum(1 for n in trainable_names if 'b_logvar' in n)
w_mu_count = sum(1 for n in trainable_names if 'W_mu.weight' in n)
w_o_weight_count = sum(1 for n in trainable_names if 'W_o.weight' in n)
w_o_bias_count = sum(1 for n in trainable_names if 'W_o.bias' in n)

print(f"\nParameter breakdown:")
print(f"  b_mu: {b_mu_count}")
print(f"  b_logvar: {b_logvar_count}")
print(f"  W_mu.weight: {w_mu_count}")
print(f"  W_o.weight: {w_o_weight_count}")
print(f"  W_o.bias: {w_o_bias_count}")
print(f"  Total: {len(trainable_names)}")

if w_mu_count > 0 and b_logvar_count > 0:
    print("\n✅ LAVA parameters properly set!")
else:
    print("\n❌ LAVA parameter setup failed!")

print("=" * 60)
