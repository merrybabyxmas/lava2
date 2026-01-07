import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("Loading model...")

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# LAVA adapter 로드
checkpoint_path = "output/conversation-LAVA-Llama-2-7b-r128-seed42/checkpoint-3000"
print(f"Loading LAVA adapter from: {checkpoint_path}")

model = PeftModel.from_pretrained(
    model, 
    checkpoint_path,
    is_trainable=True  # 🔥 중요!
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# 🔥 모든 LAVA 파라미터를 GPU로 이동
print("\nMoving LAVA parameters to GPU...")
device_counts = {"cuda": 0, "cpu": 0}
for name, param in model.named_parameters():
    if 'lava' in name.lower():
        if param.device.type == 'cpu':
            param.data = param.data.to('cuda:0')
            device_counts["cpu→cuda"] = device_counts.get("cpu→cuda", 0) + 1
        else:
            device_counts[param.device.type] += 1

print(f"Device distribution: {device_counts}")

# 🔥 Training mode 확인
model.train()
print(f"\nModel training mode: {model.training}")

# 간단한 forward-backward
print("\nRunning forward-backward pass...")
inputs = tokenizer("Hello, how are you?", return_tensors="pt", padding=True)
inputs = {k: v.to('cuda:0') for k, v in inputs.items()}

try:
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    
    print(f"Loss: {loss.item():.4f}")
    
    # Backward
    loss.backward()
    
    # b_logvar의 gradient 확인
    print("\n" + "=" * 60)
    print("Checking b_logvar gradients")
    print("=" * 60)
    
    has_grad = 0
    no_grad = 0
    zero_grad = 0
    
    for name, param in model.named_parameters():
        if 'b_logvar' in name:
            has_requires_grad = param.requires_grad
            has_gradient = param.grad is not None
            
            if has_gradient:
                grad_norm = param.grad.abs().max().item()
                
                if grad_norm < 1e-10:
                    zero_grad += 1
                    status = "🚨 ZERO gradient"
                else:
                    has_grad += 1
                    status = f"✅ Has gradient (max: {grad_norm:.8f})"
            else:
                no_grad += 1
                status = "❌ NO gradient"
            
            if no_grad < 3 or has_grad < 3:  # 처음 몇 개만 출력
                print(f"\n{name[:65]}...")
                print(f"  requires_grad: {has_requires_grad}")
                print(f"  {status}")
    
    print(f"\n" + "=" * 60)
    print(f"Summary:")
    print(f"  Has gradient: {has_grad}")
    print(f"  Zero gradient: {zero_grad}")
    print(f"  No gradient: {no_grad}")
    
    # 🔥 다른 LAVA 파라미터도 확인
    print("\n" + "=" * 60)
    print("Checking other LAVA parameters (b_mu, W_mu, W_o)")
    print("=" * 60)
    
    for param_type in ['b_mu', 'W_mu.weight', 'W_o.weight']:
        has = 0
        for name, param in model.named_parameters():
            if param_type in name and 'lava' in name.lower():
                if param.grad is not None and param.grad.abs().max() > 1e-10:
                    has += 1
                    if has <= 2:
                        print(f"{param_type}: ✅ gradient = {param.grad.abs().max():.8f}")
                    if has == 1:
                        break
        
        if has == 0:
            print(f"{param_type}: ❌ NO gradient")

except Exception as e:
    print(f"\n❌ Error during forward-backward: {e}")
    import traceback
    traceback.print_exc()

