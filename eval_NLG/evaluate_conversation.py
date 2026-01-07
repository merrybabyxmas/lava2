#!/usr/bin/env python3
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

def load_model_and_tokenizer(base_model_path, adapter_path=None):
    """모델과 토크나이저 로드"""
    print(f"Loading tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading base model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        load_in_8bit=True,  # 메모리 절약
        device_map="auto",
        trust_remote_code=True
    )
    
    if adapter_path:
        print(f"Loading adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        
        # 🔥 FIX: LAVA adapter를 GPU로 명시적 이동
        print("Moving LAVA adapters to GPU...")
        device = next(model.parameters()).device
        for name, module in model.named_modules():
            if 'lava' in name.lower():
                module.to(device)
    
    model.eval()
    print("✅ Model loaded successfully!")
    return model, tokenizer

def generate_response(model, tokenizer, instruction, max_new_tokens=256, temperature=0.7):
    """단일 응답 생성"""
    prompt = PROMPT.format(instruction=instruction)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    device = next(model.parameters()).device
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
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Evaluate conversation model')
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--adapter_path', type=str, required=True)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--test_prompt', type=str, default=None)
    
    args = parser.parse_args()
    
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter_path)
    
    if args.test_prompt:
        print(f"\n📝 Testing: {args.test_prompt}")
        response = generate_response(model, tokenizer, args.test_prompt)
        print(f"\n🤖 Response: {response}\n")
    elif args.interactive:
        interactive_mode(model, tokenizer)
    else:
        print("\n⚠️  Use --interactive or --test_prompt")

if __name__ == "__main__":
    main()
