import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import logging
import os
import sys

import torch
import torch.distributed
import torch.nn.functional as F
import transformers
from transformers import Trainer, BitsAndBytesConfig
from datasets import load_dataset, concatenate_datasets
import datasets
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

# ============================================================
# 1. PEFT 시스템 강제 주입 (LAVA 등록)
# ============================================================
import peft.utils.save_and_load
import peft.mapping
from peft.utils.peft_types import PeftType

try:
    from deepspeed.runtime.zero.config import ZeroStageEnum
    from deepspeed.runtime.fp16.loss_scaler import LossScaler
    
    safe_globals = [ZeroStageEnum, LossScaler]
    
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals(safe_globals)
        print(f"✅ Registered safe globals for DeepSpeed: {[g.__name__ for g in safe_globals]}")
except ImportError:
    # DeepSpeed가 설치되지 않은 환경이거나 경로가 다를 경우 대비
    pass



if not hasattr(PeftType, "LAVA"):
    PeftType.LAVA = "LAVA"

for lava_key in ["LAVA", PeftType.LAVA]:
    peft.utils.save_and_load.PEFT_TYPE_TO_PREFIX_MAPPING[lava_key] = "adapter_model"
    peft.mapping.PEFT_TYPE_TO_PREFIX_MAPPING[lava_key] = "adapter_model"

try:
    from peft.tuners.lava import LavaConfig, LavaModel
    for lava_key in ["LAVA", PeftType.LAVA]:
        peft.mapping.PEFT_TYPE_TO_CONFIG_MAPPING[lava_key] = LavaConfig
        peft.mapping.PEFT_TYPE_TO_TUNER_MAPPING[lava_key] = LavaModel
    print("✅ LAVA mappings successfully injected into PEFT.")
except ImportError:
    print("⚠️ Warning: Could not find LavaConfig/Model in peft.tuners.lava.")

# ============================================================
# 2. 전역 설정 및 상수
# ============================================================
IGNORE_INDEX = -100
logger = logging.getLogger(__name__)

PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# 3. Arguments 설정 (dtype 기반으로 수정)
# ============================================================
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")
    attn_implementation : Optional[str] = field(default="flash_attention_2")
    full_finetune : Optional[bool] = field(default=False)
    adapter_name_or_path: Optional[str] = field(default=None)
    init_weights: str = field(default="lora", metadata={"help": "lora | pissa | lava"})
    
    # 🔥 DType 제어 인자 추가
    base_dtype: str = field(default="bf16", metadata={"help": "fp32 | bf16 | fp16 | int8 | int4"})
    adapter_dtype: str = field(default="fp32", metadata={"help": "fp32 | bf16 | fp16"})
    
    target_modules : Optional[str] = field(default="q_proj,v_proj,k_proj,o_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_alpha : Optional[float] = field(default=32.)
    lora_dropout : Optional[float] = field(default=0.)
    
    # 하위 호환성을 위해 남겨두되 내부 로직에서는 base_dtype을 우선함
    bits: int = field(default=16) 
    
    data_path: str = field(default=None)
    sub_task: List[str] = field(default=None)
    dataset_split: str = field(default="train")
    dataset_field: List[str] = field(default=None)
    model_max_length: int = field(default=512)
    merge : Optional[bool] = field(default=False)
    lambda_vib: float = field(default=0.005)
    lambda_stab: float = field(default=0.1)

# ============================================================
# 4. Custom Trainer (Stability & VIB Logging)
# ============================================================
class StabilityLavaTrainer(Trainer):
    def __init__(self, *args, lambda_vib=0.1, lambda_stab=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_vib = lambda_vib
        self.lambda_stab = lambda_stab
        self.loss_track = {"ce_loss": 0, "const_loss": 0, "vib_loss": 0}

    def compute_loss(self, model, inputs, return_outputs=False):
        if not model.training:
            return super().compute_loss(model, inputs, return_outputs)

        sub_inputs = {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        labels = sub_inputs["labels"]

        # 2-pass 효과를 위한 배치 복제
        concat_inputs = {k: torch.cat([v, v], dim=0) for k, v in sub_inputs.items()}
        outputs = model(**concat_inputs)
        logits = outputs.logits
        
        # 1. CE Loss
        concat_labels = torch.cat([labels, labels], dim=0)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = concat_labels[..., 1:].contiguous()
        ce_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # 2. Stability Loss (Symmetric KL)
        logits1, logits2 = logits.chunk(2, dim=0)
        p = F.log_softmax(logits1, dim=-1)
        q = F.softmax(logits2, dim=-1)
        p_rev = F.log_softmax(logits2, dim=-1)
        q_rev = F.softmax(logits1, dim=-1)
        const_loss = (F.kl_div(p, q, reduction='batchmean') + F.kl_div(p_rev, q_rev, reduction='batchmean')) / 2

        # 3. VIB Loss
        kl_divs = []
        for m in model.modules():
            if hasattr(m, "_last_mu") and getattr(m, "_last_mu") is not None:
                # 첫 번째 pass의 통계치만 사용
                mu, logvar = m._last_mu.chunk(2)[0], m._last_logvar.chunk(2)[0]
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
                kl_divs.append(kl)
        vib_loss = torch.stack(kl_divs).mean() if kl_divs else torch.tensor(0.0).to(ce_loss.device)

        loss = ce_loss + self.lambda_stab * const_loss + self.lambda_vib * vib_loss
        
        if self.state.global_step % self.args.logging_steps == 0:
            self.loss_track["ce_loss"] = ce_loss.detach().item()
            self.loss_track["const_loss"] = const_loss.detach().item()
            self.loss_track["vib_loss"] = vib_loss.detach().item()
            
        return (loss, outputs) if return_outputs else loss
        
    def log(self, logs: Dict[str, float]) -> None:
        logs["train/ce_loss"] = self.loss_track["ce_loss"]
        logs["train/const_loss"] = self.loss_track["const_loss"]
        logs["train/vib_loss"] = self.loss_track["vib_loss"]
        super().log(logs)

class SavePeftModelCallback(transformers.TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def on_save(self, args, state, control, **kwargs):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        self.tokenizer.save_pretrained(peft_model_path)
        return control

# ============================================================
# 5. 모델 구축 (DType 기반 Sharding 및 Precision 설정)
# ============================================================
def build_model(script_args, checkpoint_dir):
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    
    b_type_str = script_args.base_dtype.lower()
    a_type_str = script_args.adapter_dtype.lower()
    
    # 내부 연산용 dtype (bf16/fp16/fp32 결정)
    compute_dtype = torch.bfloat16 if b_type_str in ["int4", "int8", "bf16"] else torch.float32
    
    # 양자화 설정
    q_config = None
    if b_type_str == "int4":
        q_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_quant_type="nf4")
    elif b_type_str == "int8":
        q_config = BitsAndBytesConfig(load_in_8bit=True)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=q_config,
        torch_dtype=compute_dtype if q_config else dtype_map.get(b_type_str, torch.float32),
        trust_remote_code=True,
    )

    if not script_args.full_finetune:
        if q_config is not None:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=script_args.gradient_checkpointing)

        if checkpoint_dir is not None:
            model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=True)
        elif script_args.init_weights.lower() == "lava":
            from peft.tuners.lava.config import LavaConfig
            peft_config = LavaConfig(
                r=script_args.lora_rank,
                target_modules=script_args.target_modules.split(","),
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, peft_config)
            
            # 🔥 어댑터 정밀도 강제 설정 (adapter_dtype)
            target_a_dtype = dtype_map.get(a_type_str, torch.float32)
            for name, param in model.named_parameters():
                if 'lava' in name.lower() or 'lora' in name.lower():
                    param.data = param.data.to(target_a_dtype)
                    param.requires_grad = True
        else:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=script_args.target_modules.split(','),
                r=script_args.lora_rank, 
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                init_lora_weights=script_args.init_weights,
            )
            model = get_peft_model(model, peft_config)

        if script_args.gradient_checkpointing:
            model.enable_input_require_grads()
            model.config.use_cache = False

    return model

# ============================================================
# 6. 데이터 로딩 및 실행
# ============================================================
def _tokenize_fn(strings, tokenizer):
    tokenized_list = [tokenizer(text, max_length=tokenizer.model_max_length, truncation=True) for text in strings]
    input_ids = [np.array(tokenized.input_ids) for tokenized in tokenized_list]
    return dict(input_ids=input_ids, labels=copy.deepcopy(input_ids))

def preprocess(sources, targets, tokenizer):
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized = _tokenize_fn(examples, tokenizer)
    sources_tokenized = _tokenize_fn(sources, tokenizer)
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, [len(s) for s in sources_tokenized["input_ids"]]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map(dict(instruction=instr)) for instr in examples[query]]
    targets = [f"{out}\n{tokenizer.eos_token}" for out in examples[response]]
    return preprocess(sources, targets, tokenizer)

def get_last_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir): return None
    if os.path.exists(os.path.join(checkpoint_dir, 'completed')): return None
    max_step = 0
    for filename in os.listdir(checkpoint_dir):
        if os.path.isdir(os.path.join(checkpoint_dir, filename)) and filename.startswith(PREFIX_CHECKPOINT_DIR):
            max_step = max(max_step, int(filename.replace(PREFIX_CHECKPOINT_DIR + '-', '')))
    return os.path.join(checkpoint_dir, f'{PREFIX_CHECKPOINT_DIR}-{max_step}') if max_step > 0 else None

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances):
        input_ids = [torch.tensor(instance["input_ids"]) for instance in instances]
        labels = [torch.tensor(instance["labels"]) for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))

def train():
    set_seed(42)
    parser = transformers.HfArgumentParser(TrainingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    # ---------------------------------------------------------
    # 🔥 1. 체크포인트 경로 설정 (DType 정보 포함)
    # ---------------------------------------------------------
    clean_model_name = script_args.model_name_or_path.split("/")[-1]
    dataset_name = script_args.sub_task[0].split(":")[0] if script_args.sub_task else "unknown"
    
    setting_str = (
        f"M-{clean_model_name}_"
        f"A-{script_args.init_weights}_"
        f"B-{script_args.base_dtype}_"
        f"AD-{script_args.adapter_dtype}_"
        f"R-{script_args.lora_rank}_"
        f"VIB-{script_args.lambda_vib}_"
        f"S-{script_args.seed}"
    )
    script_args.output_dir = os.path.join(script_args.output_dir, setting_str)
    script_args.run_name = setting_str
    
    # DeepSpeed용 bf16 플래그 자동 설정
    if script_args.base_dtype == "bf16":
        script_args.bf16 = True

    if script_args.local_rank == 0:
        print(f"🚀 Training starting. Output directory: {script_args.output_dir}")

    # ---------------------------------------------------------
    # 2. 모델 및 데이터 로드
    # ---------------------------------------------------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.model_name_or_path, 
        model_max_length=script_args.model_max_length, 
        padding_side="right", 
        use_fast=True
    )
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    resume_from_checkpoint_dir = get_last_checkpoint(script_args.output_dir)
    model = build_model(script_args, resume_from_checkpoint_dir)

    all_ds = []
    for task in script_args.sub_task:
        task_name, split_info = task.split(":") if ":" in task else (task, script_args.dataset_split)
        current_split = f"{script_args.dataset_split}[:{split_info}]" if ":" in task else split_info
        all_ds.append(load_dataset(script_args.data_path, data_dir=task_name, split=current_split))
    
    train_dataset = concatenate_datasets(all_ds).map(
        train_tokenize_function,
        batched=True,
        remove_columns=all_ds[0].column_names,
        fn_kwargs={"tokenizer": tokenizer, "query": script_args.dataset_field[0], "response": script_args.dataset_field[1]}
    )

    data_module = dict(train_dataset=train_dataset, data_collator=DataCollatorForSupervisedDataset(tokenizer))

    # ---------------------------------------------------------
    # 3. Trainer 실행
    # ---------------------------------------------------------
    trainer = StabilityLavaTrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=script_args, 
        lambda_vib=script_args.lambda_vib, 
        lambda_stab=script_args.lambda_stab, 
        **data_module
    )
    
    if not script_args.full_finetune:
        trainer.add_callback(SavePeftModelCallback(tokenizer))

    trainer.train(resume_from_checkpoint=resume_from_checkpoint_dir)
    trainer.save_state()
    
    if not script_args.full_finetune and script_args.merge:
        model = model.merge_and_unload()
        model.save_pretrained(script_args.output_dir)

if __name__ == "__main__":
    train()