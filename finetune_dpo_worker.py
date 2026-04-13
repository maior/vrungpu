"""
VrunGPU DPO Fine-Tuning Worker - PEFT LoRA + TRL DPOTrainer
V100 호환 fp16 DPO (LoRA 사용 시 ref_model 중복 로딩 불필요)

Dataset format (JSONL):
    {"prompt": "...", "chosen": "...", "rejected": "..."}

Usage:
    CUDA_VISIBLE_DEVICES=1 python finetune_dpo_worker.py \
        --model /path/to/model \
        --dataset /path/to/pairs.jsonl \
        --output-dir /path/to/output \
        --epochs 1 --beta 0.1 --lora-r 16
"""

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from trl.trainer.callbacks import TrainerCallback


_cancel = False


def handle_sigterm(sig, frame):
    global _cancel
    _cancel = True
    print("[FINETUNE:status=cancelling]", flush=True)


signal.signal(signal.SIGTERM, handle_sigterm)


def load_dpo_dataset(path: str) -> Dataset:
    """JSONL → datasets.Dataset (prompt/chosen/rejected)"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not all(k in item for k in ("prompt", "chosen", "rejected")):
                continue
            rows.append({
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": item["rejected"],
            })
    if not rows:
        raise ValueError("빈 데이터셋 또는 prompt/chosen/rejected 필드 누락")
    print(f"[FINETUNE:dataset_loaded={len(rows)}]", flush=True)
    return Dataset.from_list(rows)


class ProgressCallback(TrainerCallback):
    """TRL Trainer 이벤트 → VrunGPU 표준 로그 라인 변환"""

    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.total_steps = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.total_steps = state.max_steps or 0
        print(
            f"[FINETUNE:total_steps={self.total_steps},"
            f"total_epochs={self.total_epochs}]",
            flush=True,
        )
        print("[PROGRESS:20.0:Training started...]", flush=True)
        print("[FINETUNE:status=training]", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = state.global_step
        total = self.total_steps or max(step, 1)
        progress = 20 + (step / total) * 75  # 20~95%
        loss = logs.get("loss") or logs.get("train_loss")
        epoch_f = logs.get("epoch", state.epoch or 0)
        epoch = int(epoch_f) + 1 if isinstance(epoch_f, float) and epoch_f < self.total_epochs else max(1, int(epoch_f))
        loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else "n/a"
        print(
            f"[PROGRESS:{progress:.1f}:Epoch {epoch}/{self.total_epochs}, "
            f"Step {step}/{total}, Loss: {loss_str}]",
            flush=True,
        )
        finetune_fields = [
            f"epoch={epoch}",
            f"step={step}",
            f"total_steps={total}",
        ]
        if isinstance(loss, (int, float)):
            finetune_fields.append(f"loss={loss:.4f}")
        if "rewards/chosen" in logs:
            finetune_fields.append(f"reward_chosen={logs['rewards/chosen']:.4f}")
        if "rewards/rejected" in logs:
            finetune_fields.append(f"reward_rejected={logs['rewards/rejected']:.4f}")
        if "learning_rate" in logs:
            finetune_fields.append(f"lr={logs['learning_rate']:.8f}")
        print(f"[FINETUNE:{','.join(finetune_fields)}]", flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        if _cancel:
            control.should_training_stop = True


def train(args):
    start_time = time.time()
    print("[FINETUNE:status=loading_model]", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Output: {args.output_dir}", flush=True)
    print(f"Beta: {args.beta}", flush=True)
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base model (fp16)
    print("[PROGRESS:5.0:Loading model (fp16)...]", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    mem_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"Model loaded: {mem_gb:.2f} GB", flush=True)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    # transformers 5.x removed PreTrainedModel.warnings_issued, but TRL 0.24
    # still writes to model.warnings_issued["estimate_tokens"]. Pre-seed it
    # on the base model so the PeftModel getattr chain resolves.
    if not hasattr(model, "warnings_issued") or getattr(model, "warnings_issued", None) is None:
        model.warnings_issued = {}

    # LoRA (ref_model=None 으로 base 자동 공유 → VRAM 절약)
    print(f"[PROGRESS:10.0:Applying LoRA (r={args.lora_r})...]", flush=True)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules="all-linear",
    )

    # Dataset
    print("[PROGRESS:15.0:Loading dataset...]", flush=True)
    dataset = load_dpo_dataset(args.dataset)

    # DPO config
    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        beta=args.beta,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_seq_length // 2,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        logging_steps=max(1, args.logging_steps),
        save_strategy="no",  # 최종에만 수동 저장
        report_to="none",
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # LoRA가 있으면 adapter disable 로 base 자동 참조
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[ProgressCallback(total_epochs=args.epochs)],
    )

    # Save config
    config = {
        "mode": "dpo",
        "model": args.model,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "beta": args.beta,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_seq_length": args.max_seq_length,
        "dataset_size": len(dataset),
        "started_at": datetime.now().isoformat(),
    }
    (output_dir / "training_args.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False)
    )

    # Train
    result = trainer.train()

    if _cancel:
        print("[FINETUNE:status=cancelling,saving_checkpoint]", flush=True)
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(output_dir)
        config["status"] = "cancelled"
        config["elapsed_seconds"] = round(time.time() - start_time, 1)
        (output_dir / "training_args.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False)
        )
        print("[FINETUNE:status=cancelled]", flush=True)
        print("[PROGRESS:100.0:Cancelled (checkpoint saved)]", flush=True)
        sys.exit(0)

    # Final save
    print("[PROGRESS:95.0:Saving model...]", flush=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)

    elapsed = time.time() - start_time
    final_loss = float(result.training_loss) if result and hasattr(result, "training_loss") else None
    config["completed_at"] = datetime.now().isoformat()
    config["elapsed_seconds"] = round(elapsed, 1)
    config["final_loss"] = round(final_loss, 4) if final_loss is not None else None
    config["status"] = "completed"
    (output_dir / "training_args.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False)
    )

    loss_str = f"{final_loss:.4f}" if final_loss is not None else "n/a"
    print(
        f"[FINETUNE:status=completed,final_loss={loss_str},elapsed={elapsed:.1f}s]",
        flush=True,
    )
    print(f"[PROGRESS:100.0:DPO complete! Loss: {loss_str}]", flush=True)
    print(f"Model saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VrunGPU DPO Fine-Tuning Worker")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--logging-steps", type=int, default=5)
    args = parser.parse_args()

    train(args)
