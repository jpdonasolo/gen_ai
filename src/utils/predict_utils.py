import re
import torch
from tqdm import tqdm

import numpy as np
from torch.utils.data import DataLoader

from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ── Inference ──────────────────────────────────────────────────────────────────

def build_messages(question: str, image) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

def predict_batch(
    model,
    processor,
    batch: dict,
    device: torch.device,
    max_new_tokens: int = 64,
) -> list[str]:
    """Run inference on a batch of {image, question} dicts."""
    messages_list = [
        build_messages(q, img)
        for q, img in zip(batch["question"], batch["image"])
    ]

    # Apply chat template for each sample, then batch
    texts = [
        processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_list
    ]

    # Collect all images in order expected by the processor
    images = [batch["image"][i] for i in range(len(batch["image"]))]

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id
        )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    predictions = processor.batch_decode(
        output_ids[:, input_len:],
        skip_special_tokens=True,
    )
    return [p.strip().lower() for p in predictions]


# ── Normalisation ──────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """Lowercase and strip punctuation/whitespace."""
    return re.sub(r"[^\w\s]", "", text.strip().lower())


def is_yes_no(answer: str) -> bool:
    return normalise(answer) in {"yes", "no"}


def extract_yes_no(text: str) -> str:
    """Return the first match found ('yes' or 'no') in the prediction, else the raw text."""
    t = normalise(text)
    match = re.search(r"\b(yes|no)\b", flags=re.I)
    if match:
        return match.group(1)
    return t


# ── Metrics ───────────────────────────────────────────────────────────────────
def token_f1(pred: str, ref: str) -> float:
    """Token-level F1 between prediction and reference (bag of words)."""
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu_n(pred: str, ref: str, n: int) -> float:
    """Sentence BLEU with only n-grams up to n (uniform weights)."""
    weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
    return sentence_bleu(
        [ref.split()], pred.split(), weights=weights,
        smoothing_function=SmoothingFunction().method1
    )


def compute_metrics(
    predictions: list[str], references: list[str]
) -> dict[str, float]:
    yes_no_correct, yes_no_total = 0, 0
    free_em, free_f1 = [], []
    free_bleu1, free_bleu2, free_bleu3 = [], [], []

    for pred, ref in zip(predictions, references):
        ref_norm = normalise(ref)
        pred_norm = normalise(pred)

        if is_yes_no(ref_norm):
            yes_no_total += 1
            yes_no_correct += int(extract_yes_no(pred_norm) == ref_norm)
        else:
            free_em.append(int(pred_norm == ref_norm))
            free_f1.append(token_f1(pred_norm, ref_norm))
            free_bleu1.append(bleu_n(pred_norm, ref_norm, 1))
            free_bleu2.append(bleu_n(pred_norm, ref_norm, 2))
            free_bleu3.append(bleu_n(pred_norm, ref_norm, 3))

    free_total = len(free_em)
    avg = lambda lst: sum(lst) / len(lst) if lst else 0.0

    return {
        # yes/no
        "yes_no_accuracy": yes_no_correct / yes_no_total if yes_no_total else 0.0,
        "yes_no_total": yes_no_total,
        # open-ended
        "exact_match": avg(free_em),
        "macro_f1": avg(free_f1),
        "bleu1": avg(free_bleu1),
        "bleu2": avg(free_bleu2),
        "bleu3": avg(free_bleu3),
        "freeform_total": free_total,
        "total": yes_no_total + free_total,
    }

def make_compute_metrics(proc):
    def _compute(eval_pred):
        pred_ids, labels = eval_pred
        vocab_size = proc.tokenizer.vocab_size
        pred_ids = np.clip(pred_ids, 0, vocab_size - 1)  # guard against overflow
        labels = np.where(labels == -100, proc.tokenizer.pad_token_id, labels)
        predictions = proc.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        references  = proc.tokenizer.batch_decode(labels,   skip_special_tokens=True)
        metrics = compute_metrics(predictions, references)
        metrics["combined_score"] = (metrics["bleu1"] + metrics["yes_no_accuracy"]) / 2
        return metrics
    return _compute