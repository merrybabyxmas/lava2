import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import random
import numpy as np
from evaluate import load as load_metric
from peft import get_peft_model, LoraConfig
from peft.tuners.lava.config import LavaConfig
# from peft.tuners.moca import MoCAConfig
# from peft.tuners.pissa import PiSSAConfig
# from peft.tuners.dora import DoRAConfig
from configs.task_config import (
    GLUE_META,
    PISSA_TASK_CONFIG,
    DORA_TASK_CONFIG,
    LORA_TASK_CONFIG,
    MOCA_TASK_CONFIG,
    LAVA_TASK_CONFIG,
)




def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 재현성 위해 (optional)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# ----------------------------------------------------------
# Task configs per adapter
# ----------------------------------------------------------


# ==========================================================
# Build PEFT config by adapter_type
# ==========================================================
def build_adapter(adapter_type, r, alpha):
    if adapter_type == "lora":
        return LoraConfig(
            r=r, lora_alpha=alpha, target_modules=["query", "key", "value", "dense"],
            task_type="SEQ_CLS"
        )

    elif adapter_type == "dora":
        return DoRAConfig(
            r=r, alpha=alpha, target_modules=["query", "key", "value", "dense"],
            task_type="SEQ_CLS"
        )

    elif adapter_type == "pissa":
        return PiSSAConfig(
            r=r, alpha=alpha, target_modules=["query", "key", "value", "dense"],
            task_type="SEQ_CLS"
        )

    elif adapter_type == "moca":
        return MoCAConfig(
            r=r, alpha=alpha, target_modules=["query", "key", "value", "dense"],
            task_type="SEQ_CLS"
        )

    elif adapter_type == "lava":
        return LavaConfig(
            r=r, target_modules=["query_proj", "key_proj", "value_proj", "dense"],
            task_type="SEQ_CLS"
        )

    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")


# ==========================================================
# MAIN TRAIN FUNCTION
# ==========================================================
def main(args):
    task = args.task
    adapter_type = args.adapter

    meta = GLUE_META[task]
    num_labels = meta["num_labels"]
    main_metric = meta["main"]
    eval_key = meta["eval_key"]

    # -------------------------------
    # 1. Load dataset
    # -------------------------------
    print("[1] Loading dataset...")
    raw = load_dataset("glue", task)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def preprocess(batch):
        s1 = batch[meta["s1"]]
        s2 = batch[meta["s2"]] if meta["s2"] else None
        return tokenizer(
            s1, s2,
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    encoded = raw.map(preprocess, batched=True)
    encoded = encoded.rename_column("label", "labels")
    encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # -------------------------------
    # 2. Base model
    # -------------------------------
    print("[2] Loading base model...")
    base = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels
    )

    # -------------------------------
    # 3. Load hparams based on adapter + task
    # -------------------------------
    if adapter_type == "pissa":
        cfg = PISSA_TASK_CONFIG[task]
    elif adapter_type == "dora":
        cfg = DORA_TASK_CONFIG[task]
    elif adapter_type == "lora":
        cfg = LORA_TASK_CONFIG[task]
    elif adapter_type == "moca":
        cfg = MOCA_TASK_CONFIG[task]
    elif adapter_type == "lava":
        cfg = LAVA_TASK_CONFIG[task]
    else:
        raise ValueError("Unknown adapter type")

    epochs = cfg["epochs"]
    batch = cfg["batch"]
    lr = cfg["lr"]
    alpha = cfg["alpha"]

    # -------------------------------
    # 4. Create Adapter
    # -------------------------------
    print(f"[3] Creating adapter {adapter_type}...")
    peft_cfg = build_adapter(adapter_type, r=args.r, alpha=alpha)

    print("[4] Injecting adapter...")
    model = get_peft_model(base, peft_cfg)

    # -------------------------------
    # 5. Metrics
    # -------------------------------
    metric = load_metric(main_metric)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if num_labels == 1:
            preds = preds[:, 0]
        else:
            preds = preds.argmax(-1)
        return metric.compute(predictions=preds, references=labels)

    # -------------------------------
    # 6. Trainer
    # -------------------------------
    print("[5] Training...")
    args_out = TrainingArguments(
        output_dir=f"./{adapter_type}_{task}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        num_train_epochs=epochs,
        load_best_model_at_end=True,
        metric_for_best_model=main_metric,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=args_out,
        train_dataset=encoded["train"],
        eval_dataset=encoded[eval_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("[6] Evaluating...")
    result = trainer.evaluate()
    print(result)


# ==========================================================
# CLI
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--adapter", type=str, required=True)  # lava/lora/moca/pissa/dora
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    setup_seed(args.seed)

    
    main(args)
