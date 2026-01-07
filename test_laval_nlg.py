import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from peft import get_peft_model
from peft.tuners.lava.config import LavaConfig


# ============================================================
# 1. NLG 모델 선택
# ============================================================
MODEL = "mistralai/Mistral-7B-v0.1"   # 🔥 원하는 NLG 모델 이름
# 예: "mistralai/Mistral-7B-Instruct-v0.2"
#     "meta-llama/Llama-3-8B"
#     "Qwen/Qwen2-7B-Instruct"


print(f"[1] Loading tokenizer for {MODEL}")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# 2. Build a toy NLG dataset (alpaca-like)
# ============================================================
print("[2] Loading dummy instruction dataset (yizhongw/self_instruct)?")

dataset = load_dataset("yizhongw/self_instruct", split="train[:200]")

def build_prompt(ex):
    instruction = ex.get("instruction", ex.get("prompt", ""))
    output = ex.get("output", ex.get("completion", ""))

    return {"text": f"Instruction: {instruction}\nResponse: {output}"}


dataset = dataset.map(build_prompt)

def preprocess(ex):
    tokens = tokenizer(
        ex["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    tokens["labels"] = tokens["input_ids"].clone()
    return {k: v.squeeze() for k, v in tokens.items()}

dataset = dataset.map(preprocess)


# ============================================================
# 3. Load Mistral CausalLM + Lava Adapter
# ============================================================
print(f"[3] Loading base CausalLM: {MODEL}")

base = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("[4] Creating LavaConfig...")

lava_cfg = LavaConfig(
    task_type="CAUSAL_LM",
    r=8,
    target_modules=[   # mistral 구조는 q_proj/k_proj/v_proj/o_proj
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

print("[5] Injecting Lava adapter...")
model = get_peft_model(base, lava_cfg)
print(model)


# ============================================================
# 4. Trainer Setup
# ============================================================
training_args = TrainingArguments(
    output_dir="./lava_mistral_test",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    warmup_steps=50,
    fp16=True,
    logging_steps=10,
    evaluation_strategy="no",
    save_strategy="no",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)


# ============================================================
# 5. Train a bit + Generate
# ============================================================
print("[6] Training...")
trainer.train()

print("[7] Test generation...")
prompt = "Write a short poem about stars."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
    )

print(tokenizer.decode(out[0], skip_special_tokens=True))
