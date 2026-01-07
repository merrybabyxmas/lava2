#!/usr/bin/env python3
"""
Conversation Model Evaluation Script (LAVA 호환)
"""

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

def get_device(model):
    """모델의 device를 안전하게 가져오기"""
    try:
        return model.device
    except:
        # PEFT 모델의 경우
        try:
            return next(model.parameters()).device
        except:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(base_model_path, adapter_path=None):
    """모델과 토크나이저 로드"""
    print(f"Loading tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading base model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    if adapter_path:
        print(f"Loading adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    model.eval()
    print("✅ Model loaded successfully!")
    return model, tokenizer

def generate_response(model, tokenizer, instruction, max_new_tokens=256, temperature=0.7):
    """단일 응답 생성"""
    prompt = PROMPT.format(instruction=instruction)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    # LAVA 호환: device 안전하게 가져오기
    device = get_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response

def interactive_mode(model, tokenizer):
    """대화형 모드"""
    print("\n" + "="*80)
    print("🤖 Interactive Chat Mode - Type 'quit' to exit")
    print("="*80 + "\n")
    
    while True:
        instruction = input("\n🤔 Instruction: ")
        
        if instruction.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not instruction.strip():
            continue
        
        print("\n🤖 Response: ", end="", flush=True)
        try:
            response = generate_response(model, tokenizer, instruction)
            print(response)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Continuing...")

def main():
    parser = argparse.ArgumentParser(description='Evaluate conversation model')
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-hf',
                       help='Base model path')
    parser.add_argument('--adapter_path', type=str, required=True,
                       help='Path to adapter (LoRA/PiSSA/LAVA)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--test_prompt', type=str, default=None,
                       help='Single test prompt')
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(
        args.base_model, 
        args.adapter_path
    )
    
    if args.test_prompt:
        # Single test
        print(f"\n📝 Testing with: {args.test_prompt}")
        response = generate_response(model, tokenizer, args.test_prompt)
        print(f"\n🤖 Response: {response}\n")
    elif args.interactive:
        # Interactive mode
        interactive_mode(model, tokenizer)
    else:
        print("\n⚠️  Please specify --interactive or --test_prompt")
        print("\nExample usage:")
        print("  python evaluate_conversation_fixed.py --adapter_path <path> --interactive")
        print("  python evaluate_conversation_fixed.py --adapter_path <path> --test_prompt 'Hello!'")

if __name__ == "__main__":
    main()
