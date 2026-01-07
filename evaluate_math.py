#!/usr/bin/env python3
"""
GSM8K & MATH Evaluation for PEFT Models
- Automatically derives model_id from adapter_path
- Saves results directly into the adapter directory
"""

import os
import json
import torch
import re
import sys
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

# WandB
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import peft
from peft.utils import PeftType
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING, PEFT_TYPE_TO_TUNER_MAPPING, PEFT_TYPE_TO_PREFIX_MAPPING

# 1. LAVA Registration
if not hasattr(PeftType, "LAVA"):
    setattr(PeftType, "LAVA", "LAVA")

try:
    from peft.tuners.lava import LavaConfig, LavaModel
except ImportError:
    pass

for lava_key in ["LAVA", getattr(PeftType, "LAVA", None)]:
    if lava_key:
        PEFT_TYPE_TO_CONFIG_MAPPING[lava_key] = LavaConfig
        PEFT_TYPE_TO_PREFIX_MAPPING[lava_key] = "adapter_model"

print("✅ LAVA mapping manually patched in evaluation script.")

def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

def load_model(base_model_path, adapter_path):
    print_flush("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if adapter_path:
        print_flush(f"Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        
        # Synchronize LAVA layers with base layers
        for name, module in model.named_modules():
            if 'lava' in name.lower():
                parent_layer_name = name.rsplit('.', 1)[0]
                try:
                    parent_layer = dict(model.named_modules())[parent_layer_name]
                    target_device = next(parent_layer.parameters()).device
                    target_dtype = next(parent_layer.parameters()).dtype
                    module.to(device=target_device, dtype=target_dtype)
                except Exception:
                    module.to(device=model.device, dtype=model.dtype)

    model.eval()
    print_flush(f"✅ Model loaded and LAVA layers synchronized.\n")
    return model, tokenizer

# [extract_answer, normalize_answer, check_answer, generate_answer 함수들은 기존과 동일]
def extract_answer(text, dataset_name):
    text = text.strip()
    if dataset_name == "gsm8k":
        if "####" in text:
            answer = text.split("####")[-1].strip()
            numbers = re.findall(r'-?\d+\.?\d*', answer)
            return numbers[0] if numbers else answer
        else:
            lines = text.split('\n')
            for line in reversed(lines):
                numbers = re.findall(r'-?\d+\.?\d*', line)
                if numbers: return numbers[-1]
            return text
    elif dataset_name == "math":
        boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
        if boxed: return boxed[-1].strip()
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        return lines[-1] if lines else text
    return text

def normalize_answer(answer):
    return str(answer).strip().lower().replace(',', '').replace(' ', '')

def check_answer(pred, gold, dataset_name):
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    if dataset_name == "gsm8k":
        try:
            return abs(float(pred_norm) - float(gold_norm)) < 1e-3
        except:
            return pred_norm == gold_norm
    elif dataset_name == "math":
        return pred_norm == gold_norm or pred_norm in gold_norm or gold_norm in pred_norm
    return False

def generate_answer(model, tokenizer, question, max_tokens=512):
    device = next(model.parameters()).device
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer[len(prompt):].strip()

def evaluate_dataset(model, tokenizer, dataset_name, num_samples=None):
    print_flush(f"\n{'='*60}\nEvaluating {dataset_name.upper()}\n{'='*60}")
    if dataset_name == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split="test")
        answer_key = "answer"
    elif dataset_name == "math":
        dataset = load_dataset("hendrycks/math", split="test")
        answer_key = "solution"
    
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    results = []; correct = 0; total = 0
    for i, example in enumerate(tqdm(dataset), 1):
        question = example["question"]
        gold_raw = example[answer_key]
        gold_extracted = gold_raw.split("####")[-1].strip() if dataset_name == "gsm8k" else extract_answer(gold_raw, dataset_name)
        
        pred_full = generate_answer(model, tokenizer, question)
        pred_answer = extract_answer(pred_full, dataset_name)
        is_correct = check_answer(pred_answer, gold_extracted, dataset_name)
        
        if is_correct: correct += 1
        total += 1
        
        if i % 10 == 0 or is_correct:
            status = "✅" if is_correct else "❌"
            print_flush(f"\n[{i}/{len(dataset)}] {status} Acc: {correct/total*100:.2f}% | Gold: {gold_extracted} | Pred: {pred_answer}")
        
        results.append({'question': question, 'gold': gold_extracted, 'pred': pred_answer, 'correct': is_correct})
        if WANDB_AVAILABLE:
            wandb.log({f'{dataset_name}/accuracy': correct / total}, step=i)
            
    return {'accuracy': correct / total, 'correct': correct, 'total': total, 'results': results}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--adapter_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['gsm8k', 'math', 'both'], default='both')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--wandb_project', type=str, default='Math-Eval')
    args = parser.parse_args()

    # --- 자동 ID 및 경로 추출 로직 ---
    # adapter_path: /.../output/FOLDER1/FOLDER2/checkpoint-XXXX/adapter_model
    abs_adapter_path = os.path.abspath(args.adapter_path)
    path_parts = abs_adapter_path.strip(os.sep).split(os.sep)
    
    # 'adapter_model'이 마지막에 있으면 제거해서 체크포인트 폴더가 기준이 되게 함
    if path_parts[-1] == 'adapter_model':
        path_parts = path_parts[:-1]
    
    # 'output' 폴더 이후의 경로만 합쳐서 ID로 사용
    if 'output' in path_parts:
        output_idx = path_parts.index('output')
        auto_id = "/".join(path_parts[output_idx+1:])
    else:
        # output이 없으면 마지막 3개 디렉토리명 사용
        auto_id = "/".join(path_parts[-3:])
    
    print_flush(f"🆔 Auto-generated Model ID: {auto_id}")
    # ------------------------------

    if WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=auto_id,
            config={'base_model': args.base_model, 'adapter_path': args.adapter_path}
        )
    
    model, tokenizer = load_model(args.base_model, args.adapter_path)
    
    datasets_to_eval = ['gsm8k', 'math'] if args.dataset == 'both' else [args.dataset]
    all_results = {}
    
    for dataset_name in datasets_to_eval:
        result = evaluate_dataset(model, tokenizer, dataset_name, args.num_samples)
        all_results[dataset_name] = result
        
        # 결과를 adapter_path 내부에 저장
        save_dir = abs_adapter_path if os.path.isdir(abs_adapter_path) else os.path.dirname(abs_adapter_path)
        output_file = os.path.join(save_dir, f"eval_{dataset_name}_results.json")
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print_flush(f"📄 Results saved to: {output_file}")
    
    # Final Summary
    print_flush(f"\n{'='*60}\n📊 FINAL SUMMARY\n{'='*60}")
    for d, r in all_results.items():
        print_flush(f"{d.upper()}: {r['accuracy']*100:.2f}% ({r['correct']}/{r['total']})")
    
    if WANDB_AVAILABLE:
        wandb.finish()

if __name__ == "__main__":
    main()