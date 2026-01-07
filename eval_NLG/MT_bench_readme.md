# MT-Bench Evaluation Guide

MT-Bench를 사용하여 학습된 대화 모델을 평가하는 방법입니다.

## 🚀 Quick Start

### 1. 환경 설정

```bash
# FastChat 설치
bash setup_mtbench.sh

# 또는 수동 설치
pip install fschat[model_worker,webui]
git clone https://github.com/lm-sys/FastChat.git
```

### 2. OpenAI API 키 설정

```bash
export OPENAI_API_KEY='sk-your-key-here'
```

**⚠️ 중요:** MT-Bench는 GPT-4를 판정자로 사용합니다. 평가당 약 $2-5의 API 비용이 발생합니다.

### 3. 단일 모델 평가

```bash
# Python 스크립트 사용 (추천)
python eval_mtbench.py \
    --model_path output/conversation-LAVA-Llama-2-7b-r8 \
    --model_id lava-r8

# 또는 Bash 스크립트
bash run_mtbench.sh \
    output/conversation-LAVA-Llama-2-7b-r8 \
    meta-llama/Llama-2-7b-hf \
    lava-r8
```

### 4. 여러 모델 배치 평가

```bash
# eval_mtbench_batch.sh 수정
# MODELS 배열에 평가할 모델 추가

bash eval_mtbench_batch.sh
```

## 📊 평가 과정

MT-Bench 평가는 3단계로 진행됩니다:

### Step 1: 모델 답변 생성
- 80개의 멀티턴 질문에 대한 답변 생성
- 8개 카테고리: Writing, Roleplay, Reasoning, Math, Coding, Extraction, STEM, Humanities
- 소요 시간: ~10-20분 (모델 크기에 따라)

### Step 2: GPT-4 판정
- GPT-4가 각 답변을 1-10점으로 평가
- 소요 시간: ~5-10분
- 비용: ~$2-5

### Step 3: 점수 계산
- 모든 질문의 평균 점수 계산
- 카테고리별 점수 제공

## 📁 출력 파일

평가 후 다음 파일들이 생성됩니다:

```
FastChat/data/mt_bench/
├── model_answer/
│   └── lava-r8.jsonl                 # 모델의 답변
├── model_judgment/
│   └── gpt-4_single.jsonl           # GPT-4 판정 결과
└── ...

mtbench_lava-r8.txt                   # 최종 점수 요약
```

## 🎯 결과 해석

MT-Bench 점수는 1-10 범위입니다:

| 점수 | 평가 |
|------|------|
| 8.0+ | Excellent (GPT-4급) |
| 7.0-8.0 | Very Good (GPT-3.5급) |
| 6.0-7.0 | Good |
| 5.0-6.0 | Average |
| < 5.0 | Needs improvement |

### 참고 점수 (Llama-2-7B 기준)

- **Llama-2-7B-chat**: ~6.27
- **Llama-2-7B + LoRA**: ~6.5-7.0 (예상)
- **Llama-2-7B + PiSSA**: ~6.8-7.2 (예상)
- **Llama-2-7B + LAVA**: ~? (평가 필요!)

## 🔧 고급 사용법

### 커스텀 평가 설정

```bash
python eval_mtbench.py \
    --model_path output/your-model \
    --model_id your-model-id \
    --num_gpus 2 \                    # 다중 GPU
    --max_tokens 2048 \                # 더 긴 답변
    --parallel 4 \                     # 병렬 판정
    --mode pairwise                    # 쌍대 비교 모드
```

### 비용 절약 팁

1. **GPT-3.5-Turbo 사용** (정확도는 낮지만 저렴)
```bash
# FastChat/fastchat/llm_judge/gen_judgment.py 수정
# DEFAULT_JUDGE = "gpt-3.5-turbo"
```

2. **샘플 수 줄이기**
```bash
# question.jsonl에서 일부 질문만 평가
```

3. **배치 평가 시 딜레이 추가**
```bash
sleep 60  # API rate limit 회피
```

## 🐛 문제 해결

### 1. OpenAI API 에러
```bash
# API 키 확인
echo $OPENAI_API_KEY

# rate limit 에러: 딜레이 추가
--parallel 1  # 병렬도 낮추기
```

### 2. CUDA Out of Memory
```bash
# GPU 메모리 부족 시
--num_gpus 2  # GPU 수 늘리기
# 또는
--max_tokens 512  # 생성 길이 줄이기
```

### 3. 모델 로딩 실패
```bash
# adapter 경로 확인
ls -la output/your-model/

# 필요 파일: adapter_config.json, adapter_model.bin
```

## 📊 결과 비교

여러 모델 평가 후:

```bash
# 모든 결과 요약
cat mtbench_*.txt | grep -A 5 "average"

# 카테고리별 비교
python compare_results.py mtbench_lava-r8.txt mtbench_lora-r16.txt
```

## 💡 Best Practices

1. **평가 전 확인사항**
   - ✅ 모델 학습 완료 확인
   - ✅ OpenAI API 크레딧 확인 ($5-10 추천)
   - ✅ GPU 메모리 충분 (최소 16GB)

2. **일관된 평가**
   - 같은 조건으로 모든 모델 평가
   - 같은 GPT 버전 사용
   - 같은 temperature/max_tokens

3. **결과 저장**
   - 평가 설정과 함께 저장
   - 날짜/시간 기록
   - 비용 트래킹

## 📚 참고 자료

- [MT-Bench 논문](https://arxiv.org/abs/2306.05685)
- [FastChat GitHub](https://github.com/lm-sys/FastChat)
- [Chatbot Arena Leaderboard](https://chat.lmsys.org/?leaderboard)

## 🆘 도움말

```bash
# Python 스크립트 도움말
python eval_mtbench.py --help

# FastChat 공식 문서
cat FastChat/fastchat/llm_judge/README.md
```