import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_name = "bert-base-uncased"
print("[1] Loading base model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
base = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# ---------------------------------------------------------
# 1. 실제 Linear layer target_modules 자동 감지
# ---------------------------------------------------------
print("[2] Detecting Linear target modules...")
target_modules = []

for name, module in base.named_modules():
    if isinstance(module, torch.nn.Linear):
        layer_name = name.split(".")[-1]
        target_modules.append(layer_name)

target_modules = sorted(list(set(target_modules)))
print("Detected target modules:", target_modules)

# 보통 deberta-v3-base에서는 아래가 필요함:
# ['query_proj', 'key_proj', 'value_proj', 'dense']
target_modules = ['dense', 'key', 'query', 'value']
# ---------------------------------------------------------
# 2. PiSSA Config
# ---------------------------------------------------------
print("[3] Building PiSSA config...")

pissa_cfg = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=target_modules,
    task_type="SEQ_CLS",
    init_lora_weights="pissa"
)

# ---------------------------------------------------------
# 3. Adapter Inject
# ---------------------------------------------------------
print("[4] Injecting adapter...")
model = get_peft_model(base, pissa_cfg)
print("Adapter injected successfully ✔")

# ---------------------------------------------------------
# 4. Forward pass
# ---------------------------------------------------------
inputs = tokenizer("hello world", return_tensors="pt")
with torch.no_grad():
    out = model(**inputs)

print("[5] Forward pass OK ✔")
print("Logits:", out.logits)
