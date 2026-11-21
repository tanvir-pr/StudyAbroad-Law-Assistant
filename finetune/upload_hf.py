"""Upload day-16 splits to the Hugging Face Hub. Token from env only — never commit it."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ingestion.sources import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("HF_DATASET_REPO", "tanvir-pr/studyabroad-law-qa-v0"),
        help="HF dataset repo id (owner/name).",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=ROOT / "finetune" / "splits",
    )
    parser.add_argument(
        "--card",
        type=Path,
        default=ROOT / "finetune" / "DATASET_CARD.md",
    )
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--public", action="store_true", help="Create/update as public.")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit(
            "Set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) in the environment. "
            "Do not put the token in the repo."
        )

    from huggingface_hub import HfApi, login

    login(token=token)
    api = HfApi()
    private = not args.public
    api.create_repo(args.repo_id, repo_type="dataset", private=private, exist_ok=True)

    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "split_summary.json"):
        path = args.splits_dir / name
        if not path.exists():
            raise SystemExit(f"Missing split file: {path}")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=args.repo_id,
            repo_type="dataset",
        )
    if args.card.exists():
        api.upload_file(
            path_or_fileobj=str(args.card),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
    print(f"Uploaded to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
