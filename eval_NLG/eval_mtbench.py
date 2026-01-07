#!/usr/bin/env python3
"""
MT-Bench Evaluation for PEFT Models
Integrates with FastChat's MT-Bench evaluation framework
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

def check_openai_key():
    """OpenAI API 키 확인"""
    if not os.environ.get('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY not set!")
        print()
        print("MT-Bench uses GPT-4 to judge model responses.")
        print("You need an OpenAI API key to proceed.")
        print()
        print("Set it with:")
        print("  export OPENAI_API_KEY='sk-...'")
        print()
        return False
    return True

def check_fastchat():
    """FastChat 설치 확인"""
    try:
        import fastchat
        return True
    except ImportError:
        print("❌ FastChat not installed!")
        print()
        print("Install with:")
        print("  pip install fschat[model_worker,webui]")
        print()
        return False

def generate_model_config(model_path, base_model, model_id, output_dir):
    """모델 설정 파일 생성"""
    config = {
        "model_id": model_id,
        "model_path": model_path,
        "base_model": base_model,
        "adapter_type": "peft",
    }
    
    config_path = Path(output_dir) / f"{model_id}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path

def run_generation(model_path, model_id, num_gpus=1, max_tokens=1024):
    """답변 생성 단계"""
    print("━" * 60)
    print("Step 1/3: Generating model answers...")
    print("━" * 60)
    
    cmd = [
        "python", "-m", "fastchat.llm_judge.gen_model_answer",
        "--model-path", model_path,
        "--model-id", model_id,
        "--num-gpus-total", str(num_gpus),
        "--max-new-token", str(max_tokens),
        "--dtype", "float16"
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=True)
    
    print("\n✅ Model answers generated!")
    return result.returncode == 0

def run_judgment(model_id, mode="single", parallel=2):
    """GPT-4 판정 단계"""
    print("\n" + "━" * 60)
    print("Step 2/3: Generating GPT-4 judgments...")
    print("━" * 60)
    print("⏳ This may take 5-10 minutes and will cost ~$2-5 in API fees")
    print()
    
    cmd = [
        "python", "-m", "fastchat.llm_judge.gen_judgment",
        "--model-list", model_id,
        "--parallel", str(parallel),
        "--mode", mode
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=True)
    
    print("\n✅ Judgments generated!")
    return result.returncode == 0

def show_results(model_id):
    """결과 출력"""
    print("\n" + "━" * 60)
    print("Step 3/3: Computing final scores...")
    print("━" * 60)
    print()
    
    cmd = [
        "python", "-m", "fastchat.llm_judge.show_result",
        "--model-list", model_id
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(result.stdout)
    
    return result.stdout

def parse_mt_bench_score(output):
    """MT-Bench 점수 파싱"""
    lines = output.strip().split('\n')
    for line in lines:
        if 'average' in line.lower():
            parts = line.split()
            for i, part in enumerate(parts):
                try:
                    score = float(part)
                    if 1.0 <= score <= 10.0:  # MT-Bench는 1-10점
                        return score
                except ValueError:
                    continue
    return None

def main():
    parser = argparse.ArgumentParser(description='Run MT-Bench evaluation')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model or adapter')
    parser.add_argument('--base_model', type=str, 
                       default='meta-llama/Llama-2-7b-hf',
                       help='Base model path (if using adapter)')
    parser.add_argument('--model_id', type=str, required=True,
                       help='Model identifier for results')
    parser.add_argument('--num_gpus', type=int, default=1,
                       help='Number of GPUs to use')
    parser.add_argument('--max_tokens', type=int, default=1024,
                       help='Max tokens to generate')
    parser.add_argument('--parallel', type=int, default=2,
                       help='Parallel judgments')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'pairwise'],
                       help='Judgment mode')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "MT-Bench Evaluation" + " " * 24 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    print(f"📊 Model: {args.model_id}")
    print(f"📁 Path:  {args.model_path}")
    print()
    
    # Checks
    if not check_fastchat():
        return 1
    
    if not check_openai_key():
        return 1
    
    # Run pipeline
    try:
        # Step 1: Generate answers
        if not run_generation(args.model_path, args.model_id, 
                             args.num_gpus, args.max_tokens):
            print("❌ Answer generation failed!")
            return 1
        
        # Step 2: Get judgments
        if not run_judgment(args.model_id, args.mode, args.parallel):
            print("❌ Judgment generation failed!")
            return 1
        
        # Step 3: Show results
        output = show_results(args.model_id)
        
        # Save results
        result_file = Path(args.output_dir) / f"mtbench_{args.model_id}.txt"
        with open(result_file, 'w') as f:
            f.write(output)
        
        # Parse score
        score = parse_mt_bench_score(output)
        
        print("\n╔" + "═" * 58 + "╗")
        print("║" + " " * 15 + "Evaluation Complete! 🎉" + " " * 21 + "║")
        print("╚" + "═" * 58 + "╝")
        print()
        
        if score:
            print(f"🎯 MT-Bench Score: {score:.2f} / 10.00")
        
        print(f"📄 Results saved to: {result_file}")
        print()
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during evaluation: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        return 1

if __name__ == "__main__":
    sys.exit(main())