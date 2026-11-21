"""Day 20: warm up google/mt5-small on a tiny XL-Sum slice (T4/CPU-friendly)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ingestion.sources import ROOT


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
    except Exception:
        pass


def load_xlsum_slice(n: int, seed: int) -> list[dict]:
    """Prefer XL-Sum; then local `xlsum_slice.jsonl`; then human refs."""
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "csebuetnlp/xlsum",
            "english",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        rows = []
        for i, row in enumerate(ds):
            if i >= n * 5:
                break
            text = (row.get("text") or "").strip()
            summary = (row.get("summary") or "").strip()
            if len(text) < 200 or len(summary) < 20:
                continue
            rows.append({"text": text[:2000], "summary": summary[:400]})
            if len(rows) >= n:
                break
        if len(rows) >= max(8, n // 2):
            return rows[:n]
    except Exception as exc:
        print(f"XL-Sum remote skipped ({exc}); trying local slice.")

    local_slice = ROOT / "t5" / "xlsum_slice.jsonl"
    if local_slice.exists():
        rows = [json.loads(line) for line in local_slice.read_text(encoding="utf-8").splitlines() if line.strip()]
        random.Random(seed).shuffle(rows)
        return rows[:n]

    local = ROOT / "t5" / "human_refs_20.jsonl"
    rows = []
    for line in local.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows.append(
            {
                "text": row["document_excerpt"],
                "summary": " ".join(row["reference_summary"].splitlines()),
            }
        )
    random.Random(seed).shuffle(rows)
    return rows[:n]


def dry_run_loss(out_dir: Path, n: int, steps: int, seed: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    loss = 3.2
    for step in range(1, steps + 1):
        loss = max(0.8, loss * 0.9 + random.uniform(-0.05, 0.05))
        history.append({"step": step, "loss": round(loss, 4)})
    payload = {
        "status": "dry_run",
        "model": "google/mt5-small",
        "n_examples": n,
        "seed": seed,
        "note": "No full mT5 train on this pass. Re-run without --dry-run on GPU/CPU with deps.",
        "history": history,
    }
    path = out_dir / "warmup_loss.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def train_warmup(rows: list[dict], *, out_dir: Path, steps: int, seed: int, batch_size: int) -> Path:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments

    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = "google/mt5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def encode(example: dict) -> dict:
        model_inputs = tokenizer(
            "summarize: " + example["text"],
            max_length=512,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            text_target=example["summary"],
            max_length=128,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    class TinyDataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = [encode(item) for item in items]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            item = self.items[idx]
            return {k: torch.tensor(v) for k, v in item.items()}

    dataset = TinyDataset(rows)
    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=batch_size,
        max_steps=steps,
        learning_rate=5e-5,
        logging_steps=max(1, steps // 5),
        save_steps=max(steps, 1),
        report_to=[],
        seed=seed,
        fp16=torch.cuda.is_available(),
    )
    losses: list[dict] = []

    class LossCatcher:
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                losses.append({"step": int(state.global_step), "loss": round(float(logs["loss"]), 4)})

    from transformers import TrainerCallback

    class CB(TrainerCallback, LossCatcher):
        pass

    trainer = Trainer(model=model, args=args, train_dataset=dataset, callbacks=[CB()])
    trainer.train()
    save_dir = out_dir / "model"
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    payload = {
        "status": "warmed",
        "model": model_name,
        "n_examples": len(rows),
        "steps": steps,
        "seed": seed,
        "history": losses,
        "model_dir": str(save_dir),
    }
    path = out_dir / "warmup_loss.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=64, help="XL-Sum slice size")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "t5" / "runs" / "mt5_small_warmup",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    rows = load_xlsum_slice(args.n, args.seed)
    if args.dry_run:
        path = dry_run_loss(args.out_dir, n=len(rows), steps=max(5, args.steps // 3), seed=args.seed)
    else:
        path = train_warmup(
            rows,
            out_dir=args.out_dir,
            steps=args.steps,
            seed=args.seed,
            batch_size=args.batch_size,
        )
    print(json.dumps({"loss_file": str(path), "n_examples": len(rows), "dry_run": args.dry_run}, indent=2))


if __name__ == "__main__":
    main()
