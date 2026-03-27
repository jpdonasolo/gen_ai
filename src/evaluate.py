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
from tqdm import tqdm

from utils import compute_metrics, predict_batch, load_base_model, load_base_dataset, load_lora_pretrained_model
from utils.config import DEVICE
from utils.dataset import preprocess_vqa, preprocess_vqa_instruct


# ── Evaluation loop ────────────────────────────────────────────────────────────

def collate_fn(batch: list[dict]) -> dict:
    """Stack list-of-dicts into dict-of-lists (HF datasets default collator drops PIL)."""
    return {
        "image": [b["images"][0] for b in batch],
        "question": [b["prompt"] for b in batch],
        "answer": [b["completion"] for b in batch],
    }


def evaluate(model, processor, ds: datasets.Dataset, batch_size: int, max_new_tokens: int):
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )

    all_preds, all_refs, all_quests = [], [], []

    for batch in tqdm(loader, desc="Evaluating"):
        preds = predict_batch(model, processor, batch, DEVICE, max_new_tokens=max_new_tokens)
        all_preds.extend(preds)
        all_refs.extend([a[0]["content"][0]["text"].strip().lower() for a in batch["answer"]])
        all_quests.extend([a[-1]["content"][0]["text"].strip().lower() for a in batch["question"]])


    return all_preds, all_refs, all_quests


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--cache-dir", default="huggingface")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None, help="Cap dataset size (for quick runs)")
    parser.add_argument("--output", default="results/eval_results.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--add-prefix", action="store_true", help="Add instruction prefix to dataset.")
    parser.add_argument("--instruct", action="store_true", help="Whether or not to load the instruct dataset.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.checkpoint is None:
        print(f"Loading model: {args.model}")
        model, processor = load_base_model(args.model, args.cache_dir)
        model.eval()
    else:
        print(f"Loading pretrained model: {args.checkpoint} from {args.model}")
        model, processor = load_lora_pretrained_model(args.checkpoint, args.model, args.cache_dir)
        model.eval()

    print(f"Loading dataset split: {args.split} ({args.add_prefix=})")
    if args.instruct:
        ds = load_base_dataset(True, cache_dir=args.cache_dir)
        ds = ds[args.split].map(
            preprocess_vqa_instruct,
            remove_columns=ds[args.split].column_names,
            num_proc=os.cpu_count(),
            writer_batch_size=500,
        )
    else:
        ds = load_base_dataset(args.add_prefix, cache_dir=args.cache_dir)
        ds = ds[args.split].map(
            preprocess_vqa,
            remove_columns=ds[args.split].column_names,
            num_proc=os.cpu_count(),
            writer_batch_size=500,
        )

    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
        print(f"Using {len(ds)} samples")

    preds, refs, quests = evaluate(model, processor, ds, args.batch_size, args.max_new_tokens)
    metrics = compute_metrics(preds, refs)

    print("\n── Results ──────────────────────────")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    payload = {"args": vars(args), "metrics": metrics, "predictions": preds, "references": refs, "questions": quests}
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()