import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import logging
import os

import torch
import torch.distributed
import transformers
from transformers import Trainer, BitsAndBytesConfig
from datasets import load_dataset, concatenate_datasets
import datasets
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel, LoraRuntimeConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

IGNORE_INDEX = -100
logger = logging.getLogger(__name__)

PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )
    
def set_seed(seed=42):
    """
    완벽한 재현을 위한 시드 고정
    """
    # Python
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class MaxEntLavaTrainer(Trainer):
    """
    MaxEnt Trainer for LAVA (Causal LM)
    - CE (token-level LM loss) 유지
    - Lava b_logvar entropy 최대화
    """

    def __init__(
        self,
        *args,
        warmup_steps=0,
        ce_margin=0.0,
        beta_lr=1e-3,
        beta_max=10.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.warmup_steps = warmup_steps
        self.ce_margin = ce_margin
        self.beta_lr = beta_lr
        self.beta_max = beta_max

        self.register_buffer("log_beta", torch.tensor(0.0))
        self.register_buffer("ema_lm_loss", torch.tensor(0.0))
        self.ema_decay = 0.99

    @property
    def beta(self):
        return torch.clamp(self.log_beta.exp(), max=self.beta_max)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        lm_loss = outputs.loss  # already masked by IGNORE_INDEX

        # EMA 업데이트
        if self.state.global_step == 0:
            self.ema_lm_loss = lm_loss.detach()
        else:
            self.ema_lm_loss = (
                self.ema_decay * self.ema_lm_loss
                + (1 - self.ema_decay) * lm_loss.detach()
            )

        # --------------------------
        # Lava entropy
        # --------------------------
        entropy = 0.0
        cnt = 0

        for name, p in model.named_parameters():
            if "lava" in name and "b_logvar" in name:
                entropy = entropy + (0.5 * (1.0 + p)).mean()
                cnt += 1

        if cnt > 0:
            entropy = entropy / cnt

        # --------------------------
        # Lagrangian update
        # --------------------------
        if self.state.global_step > self.warmup_steps:
            constraint = lm_loss - (self.ema_lm_loss + self.ce_margin)

            # ascent on beta
            self.log_beta.data += self.beta_lr * constraint.detach()

        total_loss = lm_loss - self.beta * entropy

        # logging
        if self.state.global_step % 10 == 0:
            self.log({
                "loss/lm": lm_loss.detach(),
                "loss/entropy": entropy.detach(),
                "beta": self.beta.detach(),
                "ema_lm_loss": self.ema_lm_loss.detach(),
            })

        return (total_loss, outputs) if return_outputs else total_loss




@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # Base model or residual model setting
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B")
    attn_implementation : Optional[str] = field(default="flash_attention_2")
    # Lora or PiSSA setting
    full_finetune : Optional[bool] = field(default=False)
    adapter_name_or_path: Optional[str] = field(default=None,metadata={"help": ("Pre-initialized PiSSA adapter path; when this is not None, the following arguments are ignored."),},)
    init_weights: str = field(default="lora",metadata={"help": "lora | pissa | lava"})
    use_dora : Optional[bool] = field(default=False)
    target_modules : Optional[str] = field(default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_alpha : Optional[float] = field(default=32.)
    lora_dropout : Optional[float] = field(default=0.,metadata={"help": ("Must be set to 0 when using PiSSA."),},)
    # Quantization setting
    bits: int = field(default=4,metadata={"help": "How many bits to use."})
    double_quant: bool = field(default=True,metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4",metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    # DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    sub_task: List[str] = field(default=None)
    dataset_split: str = field(default="train", metadata={"help": "(`['train', 'test', 'eval']`):"})
    dataset_field: List[str] = field(default=None, metadata={"help": "Fields of dataset input and output."})
    shuffle_dataset : Optional[bool] = field(default=False)
    # TrainingArguments
    optim: str = field(default="paged_adamw_8bit")
    model_max_length: int = field(default=512,metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},)
    merge : Optional[bool] = field(default=False,metadata={"help": "Merge the PiSSA adapter to the residual model or LoRA to the base model"},)
    seed=42
    data_seed=42

class SavePeftModelCallback(transformers.TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def save_model(self, args, state, kwargs):
        logger.info('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        tokenizer = kwargs.get("tokenizer", self.tokenizer)
        tokenizer.save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)
        touch(os.path.join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

def get_last_checkpoint(checkpoint_dir):
    if os.path.isdir(checkpoint_dir):
        is_completed = os.path.exists(os.path.join(checkpoint_dir, 'completed'))
        if is_completed: return None # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(os.path.join(checkpoint_dir, filename)) and filename.startswith(PREFIX_CHECKPOINT_DIR):
                max_step = max(max_step, int(filename.replace(PREFIX_CHECKPOINT_DIR + '-', '')))
        if max_step == 0: return None
        latest_ckpt_dir = os.path.join(checkpoint_dir, f'{PREFIX_CHECKPOINT_DIR}-{max_step}')
        logger.info(f"Found a previous checkpoint at: {checkpoint_dir}")
        return latest_ckpt_dir
    return None # first training

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [tokenizer(text, max_length=tokenizer.model_max_length,truncation=True,)for text in strings]
    input_ids = labels = [np.array(tokenized.input_ids) for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [len(tokenized.input_ids) for tokenized in tokenized_list]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
    targets = [f"{output}\n{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def build_model(script_args, checkpoint_dir):
    if script_args.full_finetune:
        assert script_args.bits in [16, 32]
    compute_dtype = (torch.bfloat16 if script_args.bf16 else torch.float32)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=script_args.bits == 4,
            load_in_8bit=script_args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=script_args.double_quant,
            bnb_4bit_quant_type=script_args.quant_type,
        ) if script_args.bits in [4, 8] else None,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    
    if not script_args.full_finetune:
        if script_args.bits < 16:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=script_args.gradient_checkpointing)

        if checkpoint_dir is not None:
            logger.info(f"Loading adapters from {checkpoint_dir}.")
            model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=True)
            
        elif script_args.adapter_name_or_path is not None:
            logger.info(f"Initilize LoRA/PiSSA/CLOVER adapters from {script_args.model_name_or_path}/{script_args.adapter_name_or_path}.")
            model = PeftModel.from_pretrained(model, script_args.model_name_or_path, subfolder = script_args.adapter_name_or_path, is_trainable=True)
            
        elif isinstance(script_args.init_weights, str) and script_args.init_weights.lower() == "lava":
            logger.info("Init LAVA modules...")

            from peft.tuners.lava.config import LavaConfig

            peft_config = LavaConfig(
                r=script_args.lora_rank,
                target_modules=script_args.target_modules.split(","),
                task_type=TaskType.CAUSAL_LM,
            )

            model = get_peft_model(model, peft_config)
            script_args.merge = False
            
            # 🔥 LAVA 전용 처리: parameter name 기반으로 처리
            logger.info("Setting LAVA parameters to trainable...")
            
            trainable_params = []
            for name, param in model.named_parameters():
                # LAVA 파라미터만 trainable로 설정
                if 'lava' in name.lower():
                    if param.dtype.is_floating_point:
                        param.requires_grad = True
                        trainable_params.append(name)
                        
                        # 🔥 Device 이동
                        if param.device.type == 'cpu':
                            param.data = param.data.to('cuda:0')
                    else:
                        param.requires_grad = False
                else:
                    # Non-LAVA 파라미터는 freeze
                    param.requires_grad = False
            
            if script_args.local_rank == 0:
                logger.info(f"Set {len(trainable_params)} LAVA parameters to trainable")
                logger.info(f"Sample trainable params: {trainable_params[:5]}")
   
        else:
            logger.info(f'Init LoRA/PiSSA modules...')
            peft_config = LoraConfig(
                use_dora=script_args.use_dora,
                runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=script_args.use_dora),
                task_type=TaskType.CAUSAL_LM,
                target_modules=script_args.target_modules.split(','),
                inference_mode=False,
                r=script_args.lora_rank, 
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                init_lora_weights=script_args.init_weights,
            )
            model = get_peft_model(model, peft_config)
            # LoRA/PiSSA는 PEFT가 자동으로 requires_grad 처리
        
        # 🔥 LoRA/PiSSA의 경우 gradient checkpointing 수동 활성화
        if script_args.gradient_checkpointing:
            if isinstance(script_args.init_weights, str) and script_args.init_weights.lower() == "lava":
                logger.info("⚠️ LAVA는 gradient checkpointing을 지원하지 않습니다.")
            else:
                logger.info("Enabling gradient checkpointing...")
                # PEFT 모델에서 gradient checkpointing 활성화
                model.enable_input_require_grads()
            
                # base_model에 gradient checkpointing 적용
                if hasattr(model, "base_model"):
                    if hasattr(model.base_model, "model"):
                        model.base_model.model.gradient_checkpointing_enable()
                        model.base_model.model.config.use_cache = False
                    elif hasattr(model.base_model, "gradient_checkpointing_enable"):
                        model.base_model.gradient_checkpointing_enable()
                        if hasattr(model.base_model, "config"):
                            model.base_model.config.use_cache = False
    
    # 🔥 학습 가능한 파라미터 확인
    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    
    logger.info(f"Trainable params: {trainable_count:,} / {total_params:,} ({100 * trainable_count / total_params:.2f}%)")
    
    if len(trainable_params) == 0:
        logger.error("⚠️ WARNING: No trainable parameters found!")
        logger.error("This will cause training to fail. Please check your configuration.")
        raise ValueError("No trainable parameters found in the model!")
    
    if script_args.local_rank == 0:
        logger.info(f"First 20 trainable parameters:")
        for name in trainable_params[:20]:
            logger.info(f"  - {name}")
    
    return model
    
def train():
    set_seed(42)

    parser = transformers.HfArgumentParser(TrainingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    log_level = script_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
        
    if script_args.local_rank == 0:
        logger.info('='*100)
        logger.info(script_args)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        model_max_length=script_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if script_args.local_rank == 0:
        logger.info("Load tokenizer from {} over.".format(script_args.model_name_or_path))
    
    resume_from_checkpoint_dir = get_last_checkpoint(script_args.output_dir)
    model = build_model(script_args, resume_from_checkpoint_dir)

    all_training_dataset = []
    for task in script_args.sub_task:
        if ":" in task: # e.g. math:500, gsm8k:100
            cur_task, num_split = task.split(":")
            cur_split = f"{script_args.dataset_split}[:{num_split}]"
        else:
            cur_task, cur_split = task, script_args.dataset_split

        ds = load_dataset(script_args.data_path, data_dir=cur_task, split=cur_split)
        if script_args.local_rank == 0:
            print(f"{script_args.data_path}/{cur_task}/{cur_split}/{ds.num_rows}")
            for k,v in ds[0].items():
                print("-"*100)
                print(k,end=':\t')
                print(v)
            print("+"*100)
        all_training_dataset.append(ds)
        
    raw_train_datasets = concatenate_datasets(all_training_dataset)
    if script_args.shuffle_dataset:
        if script_args.local_rank == 0:
            print(f"Shuffle dataset with seed={script_args.seed}")
        raw_train_datasets = raw_train_datasets.shuffle(seed=script_args.seed)

    if script_args.local_rank > 0: 
        torch.distributed.barrier()
        
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=1,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={"tokenizer": tokenizer, "query": script_args.dataset_field[0], "response": script_args.dataset_field[1]}
    )

        
    if script_args.local_rank == 0:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        print(model)
        logger.info("Training dataset samples:", len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]['input_ids']}, {train_dataset[index]['labels']}.")
            logger.info(f"Sample {index} of the training set: {tokenizer.decode(list(train_dataset[index]['input_ids']))}.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

    trainer = Trainer(model=model, tokenizer=tokenizer, args=script_args, **data_module)
    if not script_args.full_finetune:
        trainer.add_callback(SavePeftModelCallback(tokenizer))
    trainer.train(resume_from_checkpoint = resume_from_checkpoint_dir)
    trainer.save_state()
    if not script_args.full_finetune and script_args.merge:
        model = model.merge_and_unload()
        model.save_pretrained(script_args.output_dir)
        tokenizer.save_pretrained(script_args.output_dir)
    if script_args.full_finetune:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=script_args.output_dir)
        

if __name__ == "__main__":
    train()
