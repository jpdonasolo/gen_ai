import os
import numpy as np
from torch.utils.data import Dataset
from . import load_epigraph, load_redpajama

_processor = None


# ── VQA ───────────────────────────────────────────────────────────────────────

def preprocess_vqa(example):
    img = example["image"]
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return {
        "prompt": [{"role": "user", "content": [
            {"type": "text", "text": example["question"]},
            {"type": "image"},
        ]}],
        "completion": [{"role": "assistant", "content": example["answer"]}],
        "images": [example["image"]],
    }


def make_collate_vqa(processor):
    """Return a collate function closed over *processor*."""
    def collate_vqa(examples):
        texts, images = [], []
        for example in examples:
            messages = example["prompt"] + example["completion"]
            text = processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            texts.append(text)
            images.append(example["images"][0])
        batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch
    return collate_vqa


# ── Replay ────────────────────────────────────────────────────────────────────

class ReplayDataset(Dataset):
    """Sample-level mixing: each sample is drawn from main_ds,
    or with probability aux_prob from aux_ds."""

    def __init__(self, main_ds, aux_ds, aux_prob: float = 0.1, seed: int = 42):
        self.main_ds = main_ds
        self.aux_ds = aux_ds
        self.aux_prob = aux_prob
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.main_ds)

    def __getitem__(self, idx):
        if self.rng.random() < self.aux_prob:
            aux_idx = int(self.rng.integers(len(self.aux_ds)))
            return self.aux_ds[aux_idx]
        return self.main_ds[idx]


def _preprocess_epigraph(example):
    img = example["image"]
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": example["text"]},
        ],
    }]
    text = _processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"prompt": text, "images": [img]}


def _preprocess_redpajama(example):
    return {"prompt": example["text"], "images": []}


def get_replay_dataset(processor, cache_dir: str, rp_max_len: int) -> ReplayDataset:
    global _processor
    _processor = processor

    epigraph_ds = load_epigraph(cache_dir=cache_dir)
    redpajama_ds = load_redpajama(tokenizer=processor.tokenizer, max_length=rp_max_len, cache_dir=cache_dir)

    epigraph_ds = epigraph_ds.map(
        _preprocess_epigraph, remove_columns=epigraph_ds.column_names, num_proc=os.cpu_count()
    )
    redpajama_ds = redpajama_ds.map(
        _preprocess_redpajama, remove_columns=redpajama_ds.column_names, num_proc=os.cpu_count()
    )
    return ReplayDataset(epigraph_ds, redpajama_ds)


def make_collate_replay_dataset(processor):
    """Return a collate function closed over *processor*."""
    def collate_replay_dataset(examples):
        texts = [ex["prompt"] for ex in examples]
        flat_images = [img for ex in examples for img in ex["images"]]
        batch = processor(
            text=texts,
            images=flat_images if flat_images else None,
            return_tensors="pt",
            padding=True,
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch
    return collate_replay_dataset