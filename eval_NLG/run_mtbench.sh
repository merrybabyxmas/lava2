#!/bin/bash
# MT-Bench 평가 스크립트

set -e

# ============================================================
# Configuration
# ============================================================
MODEL_PATH="${1:-output/conversation-LAVA-Llama-2-7b-r8}"
BASE_MODEL="${2:-meta-llama/Llama-2-7b-hf}"
MODEL_ID="${3:-lava-llama2-7b-r8}"

FASTCHAT_DIR="FastChat"
BENCH_NAME="mt_bench"

echo "╔════════════════════════════════════════════════════════╗"
echo "║          MT-Bench Evaluation Pipeline                  ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Configuration:"
echo "  Model Path:  $MODEL_PATH"
echo "  Base Model:  $BASE_MODEL"
echo "  Model ID:    $MODEL_ID"
echo ""

# ============================================================
# Check OpenAI API Key
# ============================================================
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set"
    echo "   MT-Bench uses GPT-4 as a judge. You need an OpenAI API key."
    echo ""
    echo "   Set it with: export OPENAI_API_KEY='your-key-here'"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# ============================================================
# Check FastChat installation
# ============================================================
if [ ! -d "$FASTCHAT_DIR" ]; then
    echo "❌ FastChat not found!"
    echo "   Run: bash setup_mtbench.sh"
    exit 1
fi

cd $FASTCHAT_DIR

# ============================================================
# Step 1: Generate model answers
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1/3: Generating model answers..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python -m fastchat.llm_judge.gen_model_answer \
    --model-path "../$MODEL_PATH" \
    --model-id "$MODEL_ID" \
    --num-gpus-total 1 \
    --max-new-token 1024 \
    --dtype float16

echo "✅ Model answers generated!"

# ============================================================
# Step 2: Generate GPT-4 judgments
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2/3: Generating GPT-4 judgments..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python -m fastchat.llm_judge.gen_judgment \
    --model-list "$MODEL_ID" \
    --parallel 2 \
    --mode single

echo "✅ Judgments generated!"

# ============================================================
# Step 3: Show results
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3/3: Computing scores..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python -m fastchat.llm_judge.show_result \
    --model-list "$MODEL_ID"

# Save results
RESULT_FILE="../mtbench_results_${MODEL_ID}.txt"
python -m fastchat.llm_judge.show_result \
    --model-list "$MODEL_ID" > "$RESULT_FILE"

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║              Evaluation Complete! 🎉                   ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Results saved to: $RESULT_FILE"
echo ""
echo "📁 Detailed outputs:"
echo "   Answers:    FastChat/data/mt_bench/model_answer/${MODEL_ID}.jsonl"
echo "   Judgments:  FastChat/data/mt_bench/model_judgment/gpt-4_single.jsonl"
echo ""

cd ..