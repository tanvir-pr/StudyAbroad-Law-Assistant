"""Day 17: QLoRA fine-tune (r=16, one seed). Saves loss curves. Target: T4 GPU."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import yaml

from ingestion.sources import ROOT


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def format_example(row: dict) -> dict:
    prompt = (
        "You are StudyAbroad Law Assistant. Answer from official Hungary study rules only. "
        "Cite sources. Not legal advice.\n\n"
        f"Question: {row['question']}\n\nAnswer:"
    )
    return {"text": prompt + " " + row["answer"]}


def write_dry_run_loss(out_dir: Path, seed: int, r: int, n_train: int) -> Path:
    """CPU / no-GPU placeholder so day-17 artifacts exist before a T4 run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = max(5, min(40, n_train // 4))
    history = []
    loss = 2.4
    for step in range(1, steps + 1):
        loss = max(0.35, loss * 0.92 + random.uniform(-0.02, 0.02))
        history.append({"step": step, "loss": round(loss, 4), "epoch": round(step / steps, 3)})
    payload = {
        "status": "dry_run",
        "note": "No CUDA QLoRA run on this machine. Re-run without --dry-run on a T4.",
        "seed": seed,
        "lora_r": r,
        "n_train": n_train,
        "history": history,
    }
    path = out_dir / "loss.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "loss.csv").write_text(
        "step,epoch,loss\n" + "\n".join(f"{h['step']},{h['epoch']},{h['loss']}" for h in history) + "\n",
        encoding="utf-8",
    )
    return path


def train(cfg: dict, *, dry_run: bool, max_steps: int | None) -> Path:
    seed = int(cfg["seed"])
    set_seed(seed)
    splits = ROOT / "finetune" / "splits"
    train_rows = [format_example(row) for row in load_jsonl(splits / "train.jsonl")]
    val_rows = [format_example(row) for row in load_jsonl(splits / "validation.jsonl")]
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    r = int(cfg["lora"]["r"])
    if dry_run:
        path = write_dry_run_loss(out_dir, seed=seed, r=r, n_train=len(train_rows))
        print(json.dumps({"mode": "dry_run", "loss_file": str(path), "n_train": len(train_rows)}, indent=2))
        return path

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for QLoRA. Use --dry-run on CPU, or run on a T4.")

    class LossLogger(TrainerCallback):
        def __init__(self) -> None:
            self.history: list[dict] = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs:
                return
            self.history.append(
                {
                    "step": int(state.global_step),
                    "epoch": round(float(state.epoch or 0), 4),
                    "loss": round(float(logs["loss"]), 6),
                }
            )

    train_cfg = cfg["train"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=bool(train_cfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if train_cfg.get("bf16", True) else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model_name = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=int(cfg["lora"]["alpha"]),
        lora_dropout=float(cfg["lora"]["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg["lora"]["target_modules"]),
    )
    model = get_peft_model(model, lora_cfg)

    max_len = int(train_cfg["max_seq_length"])

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len, padding=False)

    train_ds = Dataset.from_list(train_rows).map(tokenize, batched=True, remove_columns=["text"])
    val_ds = Dataset.from_list(val_rows).map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loss_cb = LossLogger()

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        seed=seed,
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        logging_steps=int(train_cfg["logging_steps"]),
        eval_strategy="steps",
        eval_steps=int(train_cfg["eval_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        bf16=bool(train_cfg.get("bf16", True)),
        report_to=[],
        save_total_limit=2,
        max_steps=-1 if max_steps is None else int(max_steps),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[loss_cb],
    )
    trainer.train()
    trainer.save_model(str(out_dir / "adapter"))
    tokenizer.save_pretrained(str(out_dir / "adapter"))

    payload = {
        "status": "trained",
        "seed": seed,
        "lora_r": r,
        "base_model": model_name,
        "n_train": len(train_rows),
        "n_validation": len(val_rows),
        "history": loss_cb.history,
    }
    loss_path = out_dir / "loss.json"
    loss_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "loss.csv").write_text(
        "step,epoch,loss\n"
        + "\n".join(f"{h['step']},{h['epoch']},{h['loss']}" for h in loss_cb.history)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"mode": "trained", "loss_file": str(loss_path), "adapter": str(out_dir / "adapter")}, indent=2))
    return loss_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "finetune" / "qlora.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Write placeholder loss without GPU train.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional short GPU smoke run.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, dry_run=args.dry_run, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
