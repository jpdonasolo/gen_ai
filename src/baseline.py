"""
Evaluate Qwen3.5 on the PathVQA dataset.

Usage:
    python evaluate.py \
        --split test \
        --batch_size 8 \
        --max_samples 500 \
        --output results/eval_test.json
"""

import argparse
import json
import os

import datasets
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
from tqdm import tqdm

from utils import compute_metrics, predict_batch, load_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Evaluation loop ────────────────────────────────────────────────────────────

def collate_fn(batch: list[dict]) -> dict:
    """Stack list-of-dicts into dict-of-lists (HF datasets default collator drops PIL)."""
    return {
        "image": [b["image"] for b in batch],
        "question": [b["question"] for b in batch],
        "answer": [b["answer"] for b in batch],
    }


def evaluate(model, processor, split: datasets.Dataset, batch_size: int, max_new_tokens: int):
    loader = torch.utils.data.DataLoader(
        split,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )

    all_preds, all_refs = [], []

    for batch in tqdm(loader, desc="Evaluating"):
        preds = predict_batch(model, processor, batch, DEVICE, max_new_tokens=max_new_tokens)
        all_preds.extend(preds)
        all_refs.extend([a.strip().lower() for a in batch["answer"]])

    return all_preds, all_refs


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--cache_dir", default="huggingface")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None, help="Cap dataset size (for quick runs)")
    parser.add_argument("--output", default="results/eval_results.json")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.model_name}")
    model, processor = load_model(args.model_name, args.cache_dir)
    model.eval()

    print(f"Loading dataset split: {args.split}")
    ds = datasets.load_dataset("flaviagiammarino/path-vqa", cache_dir=args.cache_dir)
    split = ds[args.split]

    if args.max_samples is not None:
        split = split.select(range(min(args.max_samples, len(split))))
        print(f"Using {len(split)} samples")

    preds, refs = evaluate(model, processor, split, args.batch_size, args.max_new_tokens)
    metrics = compute_metrics(preds, refs)

    print("\n── Results ──────────────────────────")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    payload = {"args": vars(args), "metrics": metrics, "predictions": preds, "references": refs}
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()