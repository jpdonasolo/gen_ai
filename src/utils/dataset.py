import os

import numpy as np
from torch.utils.data import Dataset

from . import load_epigraph, load_redpajama
from .predict_utils import apply_chat_template


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


# ── Replay preprocessing ───────────────────────────────────────────────────────

def _preprocess_epigraph(example):
    img = example["image"]
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": example["text"]},
    ]}]
    return {"prompt": messages, "images": [img]}


def _preprocess_redpajama(example):
    messages = [{"role": "user", "content": [{"type": "text", "text": example["text"]}]}]
    return {"prompt": messages, "images": []}


# ── Unified collate ────────────────────────────────────────────────────────────

def make_collate(processor, mask_prompt: bool = True):
    """Unified collate factory for VQA and replay training.

    All examples are expected to have the shape:
        {"prompt": list[dict], "completion": list[dict] (optional), "images": list}

    The chat template is applied at collate time.

    Args:
        processor:    HuggingFace processor (tokenizer + image processor).
        mask_prompt:  If True, suppress prompt tokens in labels so loss is
                      computed on completions only. Set False for replay/LM style.
    """
    def collate(examples):
        texts, all_images, prompt_lens = [], [], []

        for example in examples:
            prompt_text = apply_chat_template(
                processor.tokenizer, example["prompt"], add_generation_prompt=True
            )
            full_text = apply_chat_template(
                processor.tokenizer, example["prompt"] + example.get("completion", [])
            )
            prompt_len = len(processor.tokenizer.encode(prompt_text, add_special_tokens=False))

            texts.append(full_text)
            all_images.append(example.get("images", []))
            prompt_lens.append(prompt_len)

        flat_images = [img for imgs in all_images for img in imgs]
        batch = processor(
            text=texts,
            images=flat_images if flat_images else None,
            return_tensors="pt",
            padding=True,
        )

        labels = batch["input_ids"].clone()
        for i, prompt_len in enumerate(prompt_lens):
            # Always mask padding tokens (processor uses left-padding).
            labels[i, batch["attention_mask"][i] == 0] = -100
            # Optionally mask prompt tokens.
            if mask_prompt and prompt_len > 0:
                content_start = int((batch["attention_mask"][i] == 1).nonzero(as_tuple=True)[0][0])
                labels[i, content_start : content_start + prompt_len] = -100

        batch["labels"] = labels
        return batch

    return collate


# ── Replay dataset ─────────────────────────────────────────────────────────────

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


def get_replay_dataset(processor, cache_dir: str, rp_max_len: int, epigraph_k: int = 20) -> ReplayDataset:
    epigraph_ds = load_epigraph(cache_dir=cache_dir, k=epigraph_k)
    redpajama_ds = load_redpajama(tokenizer=processor.tokenizer, max_length=rp_max_len, cache_dir=cache_dir)

    epigraph_ds = epigraph_ds.map(
        _preprocess_epigraph,
        remove_columns=epigraph_ds.column_names,
        num_proc=os.cpu_count(),
    )
    redpajama_ds = redpajama_ds.map(
        _preprocess_redpajama,
        remove_columns=redpajama_ds.column_names,
        num_proc=os.cpu_count(),
    )
    return ReplayDataset(epigraph_ds, redpajama_ds)