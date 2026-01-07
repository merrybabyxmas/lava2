#!/bin/bash
# MT-Bench 환경 설정 스크립트

echo "🔧 Setting up MT-Bench..."

# 1. FastChat 설치
pip install fschat[model_worker,webui] --break-system-packages

# 2. MT-Bench 데이터 다운로드
git clone https://github.com/lm-sys/FastChat.git
cd FastChat

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Set your OpenAI API key: export OPENAI_API_KEY='your-key'"
echo "2. Run evaluation: bash run_mtbench.sh"