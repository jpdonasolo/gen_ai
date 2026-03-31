import os

import numpy as np
from torch.utils.data import Dataset

from datasets import concatenate_datasets

from . import load_entigraph, load_redpajama
from .predict_utils import apply_chat_template
from .loader import load_base_dataset

from .prompts import SYSTEM_PROMPT_ENTIGRAPH


# ── VQA ───────────────────────────────────────────────────────────────────────

def preprocess_vqa(example):
    img = example["image"]
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": example["question"]},
            {"type": "image"},
        ]}]
    completion = [
        {"role": "assistant", "content": [
            {"type": "text", "text": example["answer"]}
        ]}]
    return {"prompt": messages, "images": [img], "completion": completion}

def preprocess_vqa_instruct(example):
    system_prompt = "you are a helpful assistant. answer yes-no questions only with yes or no. answer open questions with the least words possible."
    system_message = [
        {"role": "system", "content": [
            {"type": "text", "text": system_prompt}
        ]}
    ]
    ret = preprocess_vqa(example)
    ret["prompt"] = system_message + ret["prompt"]
    return ret

# ── Replay preprocessing ───────────────────────────────────────────────────────

def _preprocess_entigraph(example):
    img = example["image"]
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": example["text"]},
    ]}]
    return {"prompt": messages, "images": [img], "completion": []}

def _preprocess_entigraph_instruct(example):
    entity_lines = "\n".join(f"- {e}" for e in example["entities"])
    user_prompt = f"### Caption:\n{example['caption']}\n\n### Entities:\n{entity_lines}\n"

    img = example["image"]
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": SYSTEM_PROMPT_ENTIGRAPH}
        ]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_prompt},
        ]},
    ]
    completion = [
        {"role": "assistant", "content": [
            {"type": "text", "text": example["text"]}
        ]}
    ]
    return {"prompt": messages, "images": [img], "completion": completion}


def _preprocess_redpajama(example):
    messages = [{"role": "user", "content": [{"type": "text", "text": example["text"]}]}]
    return {"prompt": messages, "images": [], "completion": []}


# ── Unified collate ────────────────────────────────────────────────────────────
def make_collate(processor, mask_prompt: bool = True, max_length: int | None = None):

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
            labels[i, batch["attention_mask"][i] == 0] = -100
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


def get_replay_dataset(
    processor, 
    cache_dir: str, 
    rp_max_len: int, 
    entigraph_k: int = 20,
    merge_with_vqa: bool = False,
    use_aux_ds: bool = True,
    instruct: bool = False,
    load_entigraph_full: bool = False,
    max_len: int | None = None,
) -> ReplayDataset:
    num_proc = os.cpu_count()
    # To prevent memory crashes
    if load_entigraph_full:
        num_proc = num_proc // 2
    
    main_ds = load_entigraph(cache_dir=cache_dir, k=entigraph_k, load_full=load_entigraph_full, processor=processor, max_len=max_len)
    if instruct:
        main_ds = main_ds.map(
            _preprocess_entigraph_instruct,
            remove_columns=main_ds.column_names,
            num_proc=num_proc,
        )

    else:
        main_ds = main_ds.map(
            _preprocess_entigraph,
            remove_columns=main_ds.column_names,
            num_proc=num_proc,
        )

    if merge_with_vqa:
        if instruct:
            vqa_ds = load_base_dataset(add_prefixes=True, cache_dir=cache_dir)
            vqa_ds = vqa_ds["train"].map(
                preprocess_vqa_instruct,
                remove_columns=vqa_ds["train"].column_names,
                num_proc=num_proc,
                writer_batch_size=500,
            )
        else:
            vqa_ds = load_base_dataset(cache_dir=cache_dir)
            vqa_ds = vqa_ds["train"].map(
                preprocess_vqa,
                remove_columns=vqa_ds["train"].column_names,
                num_proc=num_proc,
                writer_batch_size=500,
            )
        main_ds = concatenate_datasets([main_ds, vqa_ds])

    if use_aux_ds:
        aux_ds = load_redpajama(tokenizer=processor.tokenizer, max_length=rp_max_len, cache_dir=cache_dir)
        aux_ds = aux_ds.map(
            _preprocess_redpajama,
            remove_columns=aux_ds.column_names,
            num_proc=num_proc,
        )
    else:
        aux_ds = main_ds
    
    print(f"Loaded dataset with {len(main_ds)} samples")

    return ReplayDataset(main_ds, aux_ds)