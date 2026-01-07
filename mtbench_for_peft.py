#!/usr/bin/env python3
"""
MT-Bench for PEFT Models with WandB logging
- Automatically derives model_id from adapter_path
- Saves results directly into the adapter directory
"""

import os
import json
import torch
import time
import sys
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# WandB
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  WandB not available. Install with: pip install wandb")

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

# MT-Bench 80 questions
MT_BENCH_QUESTIONS_URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"

def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

def load_model(base_model_path, adapter_path):
    print_flush("Loading model...")
    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        load_in_8bit=True,
        device_map={"": 0}
    )
    
    if adapter_path:
        print_flush(f"Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        
        device = torch.device("cuda:0")
        for name, module in model.named_modules():
            if 'lava' in name.lower():
                module.to(device)
        
        for name, param in model.named_parameters():
            if 'lava' in name.lower() and param.dtype == torch.float32:
                param.data = param.data.half()
    
    model.eval()
    load_time = time.time() - start_time
    print_flush(f"✅ Model loaded in {load_time:.2f}s\n")
    return model, tokenizer

def download_questions():
    import urllib.request
    questions_file = "mt_bench_questions.jsonl"
    if not os.path.exists(questions_file):
        print_flush("Downloading MT-Bench questions...")
        urllib.request.urlretrieve(MT_BENCH_QUESTIONS_URL, questions_file)
    
    questions = []
    with open(questions_file, 'r') as f:
        for line in f:
            questions.append(json.loads(line))
    return questions

def generate_answer(model, tokenizer, question, max_tokens=512):
    device = torch.device("cuda:0")
    inputs = tokenizer(question, return_tensors="pt").to(device)
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    generation_time = time.time() - start_time
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    num_tokens = len(outputs[0])
    tokens_per_sec = num_tokens / generation_time if generation_time > 0 else 0
    return answer, generation_time, num_tokens, tokens_per_sec

def judge_with_gpt4(question, answer):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    prompt = f"""Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n[Question]\n{question}\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]"""
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        api_time = time.time() - start_time
        judgment = response.choices[0].message.content
        import re
        match = re.search(r'\[\[(\d+)\]\]', judgment)
        score = int(match.group(1)) if match else None
        return score, judgment, api_time
    except Exception as e:
        return None, str(e), time.time() - start_time

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--adapter_path', type=str, required=True)
    parser.add_argument('--num_questions', type=int, default=80)
    parser.add_argument('--wandb_project', type=str, default='MT-Bench-Eval')
    args = parser.parse_args()
    
    if not os.environ.get('OPENAI_API_KEY'):
        print_flush("❌ OPENAI_API_KEY not set!")
        return

    # --- 자동 ID 및 경로 추출 로직 ---
    abs_adapter_path = os.path.abspath(args.adapter_path)
    path_parts = abs_adapter_path.strip(os.sep).split(os.sep)
    
    if path_parts[-1] == 'adapter_model':
        path_parts = path_parts[:-1]
    
    if 'output' in path_parts:
        output_idx = path_parts.index('output')
        auto_id = "/".join(path_parts[output_idx+1:])
    else:
        auto_id = "/".join(path_parts[-3:])
    
    # 저장 파일 경로 설정 (adapter_path 내부에 저장)
    save_dir = abs_adapter_path if os.path.isdir(abs_adapter_path) else os.path.dirname(abs_adapter_path)
    output_file = os.path.join(save_dir, "mtbench_results.json")
    
    print_flush(f"🆔 Auto-generated Model ID: {auto_id}")
    print_flush(f"💾 Results will be saved to: {output_file}")
    # ------------------------------
    
    if WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=auto_id,
            config=vars(args)
        )
    
    total_start_time = time.time()
    model, tokenizer = load_model(args.base_model, args.adapter_path)
    questions = download_questions()[:args.num_questions]
    
    results = []; scores = []; scores_by_category = {}
    total_generation_time = 0; total_api_time = 0; total_tokens = 0
    
    print_flush(f"\n🚀 Starting evaluation: {len(questions)} questions")
    
    for i, q in enumerate(questions, 1):
        question_text = q['turns'][0]
        category = q['category']
        
        print_flush(f"\n[{i}/{len(questions)}] ID: {q['question_id']} | Cat: {category}")
        
        answer, gen_time, num_tokens, tokens_per_sec = generate_answer(model, tokenizer, question_text)
        score, judgment, api_time = judge_with_gpt4(question_text, answer)
        
        total_generation_time += gen_time
        total_tokens += num_tokens
        total_api_time += api_time
        
        if score:
            scores.append(score)
            if category not in scores_by_category: scores_by_category[category] = []
            scores_by_category[category].append(score)
        
        current_avg = sum(scores) / len(scores) if scores else 0
        
        results.append({
            'question_id': q['question_id'], 'category': category,
            'question': question_text, 'answer': answer,
            'score': score, 'judgment': judgment,
            'generation_time': gen_time, 'tokens_per_sec': tokens_per_sec
        })

        # 실시간 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'model_id': auto_id,
                'progress': f"{i}/{len(questions)}",
                'last_updated': datetime.now().isoformat(),
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        print_flush(f"⭐ Score: {score}/10 | Avg: {current_avg:.2f}")

        if WANDB_AVAILABLE:
            wandb.log({'score': score, 'avg_score': current_avg, f'score/{category}': score}, step=i)
    
    total_time = time.time() - total_start_time
    avg_score = sum(scores) / len(scores) if scores else 0
    
    # 최종 요약 업데이트
    final_data = {
        'model_id': auto_id,
        'summary': {
            'avg_score': avg_score,
            'total_time': str(timedelta(seconds=int(total_time))),
            'avg_tokens_per_sec': total_tokens / total_generation_time if total_generation_time > 0 else 0
        },
        'results': results
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    print_flush(f"\n🎉 Complete! Final Avg: {avg_score:.2f} | Path: {output_file}")
    if WANDB_AVAILABLE: wandb.finish()

if __name__ == "__main__":
    main()