#!/bin/bash
# MT-Bench 실행 전 체크리스트

echo "🔍 MT-Bench 준비 상태 확인..."
echo ""

READY=true

# 1. FastChat 설치 확인
echo "1️⃣  Checking FastChat installation..."
if python3 -c "import fastchat" 2>/dev/null; then
    echo "   ✅ FastChat installed"
else
    echo "   ❌ FastChat NOT installed"
    echo "      Run: pip install fschat[model_worker,webui]"
    READY=false
fi

# 2. OpenAI API 키 확인
echo ""
echo "2️⃣  Checking OpenAI API key..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "   ❌ OPENAI_API_KEY not set"
    echo "      Run: export OPENAI_API_KEY='sk-...'"
    READY=false
else
    echo "   ✅ API key is set"
    echo "      Key: ${OPENAI_API_KEY:0:10}...${OPENAI_API_KEY: -4}"
fi

# 3. FastChat 디렉토리 확인
echo ""
echo "3️⃣  Checking FastChat directory..."
if [ -d "FastChat" ]; then
    echo "   ✅ FastChat directory exists"
    
    # question.jsonl 확인
    if [ -f "FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl" ]; then
        QUESTIONS=$(wc -l < FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl)
        echo "      Found $QUESTIONS questions"
    else
        echo "   ⚠️  Question file not found (will be downloaded on first run)"
    fi
else
    echo "   ❌ FastChat directory not found"
    echo "      Run: git clone https://github.com/lm-sys/FastChat.git"
    READY=false
fi

# 4. 모델 존재 확인
echo ""
echo "4️⃣  Checking trained models..."
for model_dir in output/conversation-*; do
    if [ -d "$model_dir" ]; then
        echo "   ✅ Found: $model_dir"
        
        # adapter_config.json 확인
        if [ -f "$model_dir/adapter_config.json" ]; then
            echo "      • adapter_config.json ✓"
        else
            echo "      ⚠️  adapter_config.json missing"
        fi
        
        # adapter_model.bin 또는 adapter_model.safetensors 확인
        if [ -f "$model_dir/adapter_model.bin" ] || [ -f "$model_dir/adapter_model.safetensors" ]; then
            echo "      • adapter weights ✓"
        else
            echo "      ⚠️  adapter weights missing"
        fi
    fi
done

# 5. GPU 확인
echo ""
echo "5️⃣  Checking GPU availability..."
if nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "   ✅ $GPU_COUNT GPU(s) available"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | \
        awk '{print "      •", $0}'
else
    echo "   ❌ nvidia-smi not found"
    READY=false
fi

# 6. 디스크 공간 확인
echo ""
echo "6️⃣  Checking disk space..."
AVAILABLE=$(df -h . | awk 'NR==2 {print $4}')
echo "   Available space: $AVAILABLE"

# 최종 판정
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ "$READY" = true ]; then
    echo "✅ All checks passed! Ready to run MT-Bench"
    echo ""
    echo "To evaluate your model, run:"
    echo "  bash run_mtbench.sh output/conversation-LAVA-Llama-2-7b-r8"
    echo ""
    echo "⚠️  This will cost ~\$2-5 in OpenAI API fees"
else
    echo "❌ Some checks failed. Please fix the issues above."
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"