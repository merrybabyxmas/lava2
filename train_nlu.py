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
import wandb

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


# ----------------------------------------------------------
# SEED SETUP
# ----------------------------------------------------------
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




from transformers import TrainerCallback
import torch

class LavaGradCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]

        stats = {
            "grad/W_mu_norm": [],
            "grad/W_o_norm": [],
            "grad/b_mu_norm": [],
            "grad/b_logvar_norm": [],
            "grad/W_mu_mean": [],
            "grad/W_o_mean": [],
            "grad/b_mu_mean": [],
            "grad/b_logvar_mean": [],
        }

        for name, p in model.named_parameters():
            if p.grad is None:
                continue

            g = p.grad.detach()

            # LavaAdapter 파라미터만 집계
            if "lava" not in name:
                continue

            if "W_mu" in name:
                stats["grad/W_mu_norm"].append(g.norm().item())
                stats["grad/W_mu_mean"].append(g.abs().mean().item())

            elif "W_o" in name:
                stats["grad/W_o_norm"].append(g.norm().item())
                stats["grad/W_o_mean"].append(g.abs().mean().item())

            elif "b_mu" in name:
                stats["grad/b_mu_norm"].append(g.norm().item())
                stats["grad/b_mu_mean"].append(g.abs().mean().item())

            elif "b_logvar" in name:
                stats["grad/b_logvar_norm"].append(g.norm().item())
                stats["grad/b_logvar_mean"].append(g.abs().mean().item())

        # 평균 내서 wandb에 기록
        log_dict = {}
        for k, v in stats.items():
            if len(v) > 0:
                log_dict[k] = sum(v) / len(v)

        if len(log_dict) > 0:
            wandb.log(log_dict, step=state.global_step)



# ----------------------------------------------------------
# Build adapter config
# ----------------------------------------------------------
def build_adapter(adapter_type, r, alpha):
    at = adapter_type.lower()

    # LoRA
    if at in ["lora", "lora_init"]:
        return LoraConfig(
            r=r, 
            lora_alpha=alpha,
            target_modules=["query_proj", "key_proj", "value_proj", "dense"],
            task_type="SEQ_CLS"
        )

    # PiSSA uses LoRAConfig but with different init scheme (pissa_init)
    if at in ["pissa", "pissa_init"]:
        return LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["query_proj", "key_proj", "value_proj", "dense"],
            task_type="SEQ_CLS",
            init_lora_weights="pissa"  # <-- PiSSA 핵심
        )

    # LAVA
    if at in ["lava", "lava_init"]:
        return LavaConfig(
            r=r,
            target_modules=["query_proj", "key_proj", "value_proj", "dense"],
            task_type="SEQ_CLS"
        )

    raise ValueError(f"Unknown adapter type: {adapter_type}")


# ----------------------------------------------------------
# MAIN TRAIN FUNCTION
# ----------------------------------------------------------
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
    # 3. Load hyperparams
    # -------------------------------
    at = adapter_type.lower()

    if at in ["pissa", "pissa_init"]:
        cfg = PISSA_TASK_CONFIG[task]

    elif at in ["dora", "dora_init"]:
        cfg = DORA_TASK_CONFIG[task]

    elif at in ["lora", "lora_init"]:
        cfg = LORA_TASK_CONFIG[task]

    elif at in ["lava", "lava_init"]:
        cfg = LAVA_TASK_CONFIG[task]

    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")

    epochs = cfg["epochs"]
    lr = args.learning_rate if args.learning_rate is not None else cfg["lr"]
    batch = args.batch if args.batch is not None else cfg["batch"]
    alpha = args.alpha if args.alpha is not None else cfg.get("alpha", None)


    # -------------------------------
    # 4. PEFT Config
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
    # 6. wandb Setup
    # -------------------------------
    model_name = args.model.split("/")[-1]
    run_name = f"{adapter_type}_{task}_seed{args.seed}_{model_name}_r{args.r}"

    wandb.init(
        project="GLUE-nlu-final",
        name=run_name,
        config={
            "task": task,
            "adapter": adapter_type,
            "seed": args.seed,
            "learning_rate": lr,
            "batch": batch,
            "epochs": epochs,
            "r": args.r,
            "alpha": alpha,
            "model": args.model
        }
    )

    # -------------------------------
    # 7. Trainer
    # -------------------------------
    print("[5] Training...")
    args_out = TrainingArguments(
        output_dir=f"./{adapter_type}_{task}",
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        num_train_epochs=epochs,
        load_best_model_at_end=False,
        metric_for_best_model=main_metric,
        report_to="wandb",
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
        callbacks=[LavaGradCallback()],
    )



    trainer.train()
    # -------------------------------
    # Compute BEST eval accuracy
    # -------------------------------
    best_acc = None
    for log in trainer.state.log_history:
        if "eval_accuracy" in log:
            acc = log["eval_accuracy"]
            if best_acc is None or acc > best_acc:
                best_acc = acc

    if best_acc is not None:
        print(f"[BEST] Best Accuracy during training: {best_acc:.4f}")
        wandb.log({"best_eval_accuracy": best_acc})

    print("[6] Evaluating...")
    result = trainer.evaluate()

    wandb.log({"final_eval": result})
    


    wandb.log({"final_eval": result})
    wandb.finish()

    print(result)


# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--alpha", type=int, default=None)




    args = parser.parse_args()
    setup_seed(args.seed)

    main(args)
