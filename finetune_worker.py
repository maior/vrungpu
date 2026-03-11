"""
VrunGPU Fine-Tuning Worker - PEFT LoRA Training
V100 호환 fp16 LoRA 파인튜닝 (QLoRA 4-bit 미사용)

Usage:
    CUDA_VISIBLE_DEVICES=1 python finetune_worker.py \
        --model /path/to/model \
        --dataset /path/to/data.jsonl \
        --output-dir /path/to/output \
        --epochs 3 --lora-r 16
"""

import argparse
import json
import math
import os
import signal
import sys
import time
import torch
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# Graceful Shutdown
# ============================================================================

_cancel = False

def handle_sigterm(sig, frame):
    global _cancel
    _cancel = True
    print("[FINETUNE:status=cancelling]", flush=True)

signal.signal(signal.SIGTERM, handle_sigterm)

# ============================================================================
# Dataset
# ============================================================================

class ChatDataset(Dataset):
    """JSONL 데이터셋 자동 감지 및 토크나이즈"""

    def __init__(self, path: str, tokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                text = self._format(item)
                if text:
                    self.examples.append(text)

        print(f"[FINETUNE:dataset_loaded={len(self.examples)}]", flush=True)

    def _format(self, item: dict) -> str | None:
        """JSONL 포맷 자동 감지"""
        # Format 1: messages (chat)
        if "messages" in item:
            try:
                return self.tokenizer.apply_chat_template(
                    item["messages"], tokenize=False, add_generation_prompt=False
                )
            except Exception:
                return None

        # Format 2: instruction/input/output (Alpaca)
        if "instruction" in item:
            messages = []
            if item.get("input"):
                messages.append({"role": "user", "content": f"{item['instruction']}\n{item['input']}"})
            else:
                messages.append({"role": "user", "content": item["instruction"]})
            if item.get("output"):
                messages.append({"role": "assistant", "content": item["output"]})
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            except Exception:
                return None

        # Format 3: raw text
        if "text" in item:
            return item["text"]

        return None

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }

# ============================================================================
# Training
# ============================================================================

def train(args):
    global _cancel

    start_time = time.time()
    print(f"[FINETUNE:status=loading_model]", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Output: {args.output_dir}", flush=True)
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')}", flush=True)

    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (fp16 for V100)
    print(f"[PROGRESS:5.0:Loading model (fp16)...]", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    mem_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"Model loaded: {mem_gb:.2f} GB", flush=True)

    # Enable gradient checkpointing (saves VRAM)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Apply LoRA
    print(f"[PROGRESS:10.0:Applying LoRA (r={args.lora_r})...]", flush=True)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print(f"[PROGRESS:15.0:Loading dataset...]", flush=True)
    dataset = ChatDataset(args.dataset, tokenizer, args.max_seq_length)
    if len(dataset) == 0:
        print("[FINETUNE:status=failed,error=empty_dataset]", flush=True)
        sys.exit(1)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # Optimizer & Scheduler & AMP Scaler (fp16 overflow 방지)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(dataloader) * args.epochs
    warmup_steps = min(100, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda")

    print(f"[FINETUNE:total_steps={total_steps},total_epochs={args.epochs},dataset_size={len(dataset)}]", flush=True)

    # Save training config
    config = {
        "model": args.model,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "total_steps": total_steps,
        "dataset_size": len(dataset),
        "started_at": datetime.now().isoformat(),
    }
    with open(output_dir / "training_args.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Training loop
    print(f"[PROGRESS:20.0:Training started...]", flush=True)
    print(f"[FINETUNE:status=training]", flush=True)

    model.train()
    global_step = 0
    log_entries = []
    best_loss = float("inf")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, batch in enumerate(dataloader):
            if _cancel:
                print(f"[FINETUNE:status=cancelling,saving_checkpoint]", flush=True)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                print(f"[FINETUNE:status=cancelled]", flush=True)
                print(f"[PROGRESS:100.0:Cancelled (checkpoint saved)]", flush=True)
                _save_log(output_dir, log_entries, config, "cancelled")
                sys.exit(0)

            # Move to GPU
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            labels = batch["labels"].to("cuda")

            # Forward + backward (AMP for fp16 stability)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            epoch_steps += 1
            step_loss = loss.item()
            if not math.isnan(step_loss) and not math.isinf(step_loss):
                epoch_loss += step_loss

            # Log entry
            log_entries.append({
                "epoch": epoch + 1,
                "step": global_step,
                "loss": round(step_loss, 4),
                "lr": round(scheduler.get_last_lr()[0], 8),
            })

            # Progress reporting (every 10 steps or 5%)
            if global_step % max(1, total_steps // 20) == 0 or global_step == total_steps:
                progress = 20 + (global_step / total_steps) * 75  # 20~95%
                avg_loss = epoch_loss / epoch_steps
                print(
                    f"[PROGRESS:{progress:.1f}:Epoch {epoch+1}/{args.epochs}, "
                    f"Step {global_step}/{total_steps}, Loss: {avg_loss:.4f}]",
                    flush=True,
                )
                print(
                    f"[FINETUNE:epoch={epoch+1},step={global_step},"
                    f"loss={avg_loss:.4f},total_steps={total_steps},"
                    f"lr={scheduler.get_last_lr()[0]:.8f}]",
                    flush=True,
                )

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"Epoch {epoch+1}/{args.epochs} complete. Avg loss: {avg_epoch_loss:.4f}", flush=True)

        # Save best checkpoint
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss

    # Save final model
    print(f"[PROGRESS:95.0:Saving model...]", flush=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    elapsed = time.time() - start_time
    config["completed_at"] = datetime.now().isoformat()
    config["elapsed_seconds"] = round(elapsed, 1)
    config["final_loss"] = round(best_loss, 4) if math.isfinite(best_loss) else None
    config["status"] = "completed"
    _save_log(output_dir, log_entries, config, "completed")

    print(f"[FINETUNE:status=completed,final_loss={best_loss:.4f},elapsed={elapsed:.1f}s]", flush=True)
    print(f"[PROGRESS:100.0:Training complete! Loss: {best_loss:.4f}]", flush=True)
    print(f"Model saved to: {output_dir}", flush=True)


def _save_log(output_dir: Path, log_entries: list, config: dict, status: str):
    """학습 로그 및 최종 설정 저장"""
    config["status"] = status
    with open(output_dir / "training_args.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with open(output_dir / "training_log.jsonl", "w") as f:
        for entry in log_entries:
            f.write(json.dumps(entry) + "\n")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VrunGPU Fine-Tuning Worker")
    parser.add_argument("--model", type=str, required=True, help="Base model path")
    parser.add_argument("--dataset", type=str, required=True, help="JSONL dataset path")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=512)
    args = parser.parse_args()

    train(args)
