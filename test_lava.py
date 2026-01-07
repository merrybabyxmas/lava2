#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys

def main():
    base_model = "meta-llama/Llama-2-7b-hf"
    adapter_path = "output/conversation-LAVA-Llama-2-7b-r8/checkpoint-2234/adapter_model"
    
    print("🔄 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        device_map="auto"
    )
    
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Fix device and dtype
    device = next(model.parameters()).device
    for name, module in model.named_modules():
        if 'lava' in name.lower():
            module.to(device)
    
    for name, param in model.named_parameters():
        if 'lava' in name.lower() and param.dtype == torch.float32:
            param.data = param.data.half()
    
    model.eval()
    
    print(f"✅ Ready on {device}!\n")
    print("="*60)
    print("💬 LAVA Chat - Type 'quit' to exit")
    print("="*60)
    
    while True:
        try:
            q = input("\n�� You: ")
        except EOFError:
            print("\n👋 Goodbye!")
            break
            
        if q.lower().strip() in ['quit', 'q', 'exit']:
            print("\n👋 Goodbye!")
            break
            
        if not q.strip():
            continue
        
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{q}\n\n### Response:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        print("\n🤖 LAVA: ", end="", flush=True)
        
        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            resp = tokenizer.decode(out[0], skip_special_tokens=True)
            resp = resp.split("### Response:")[-1].strip()
            print(resp)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
