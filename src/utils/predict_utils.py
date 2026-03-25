import re
import torch
import numpy as np
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# ── Chat template ──────────────────────────────────────────────────────────────

def apply_chat_template(tokenizer, messages, add_generation_prompt: bool = False) -> str:
    """Apply chat template with thinking disabled.
    """
    kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


# ── Inference ──────────────────────────────────────────────────────────────────

def build_messages(question: str, image) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
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
    messages_list = [
        build_messages(q, img)
        for q, img in zip(batch["question"], batch["image"])
    ]

    texts = [
        apply_chat_template(processor.tokenizer, msgs, add_generation_prompt=True)
        for msgs in messages_list
    ]

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
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    predictions = processor.batch_decode(
        output_ids[:, input_len:],
        skip_special_tokens=True,
    )
    return [p.strip().lower() for p in predictions]


# ── Normalisation ──────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.strip().lower())


def is_yes_no(answer: str) -> bool:
    return normalise(answer) in {"yes", "no"}


def extract_yes_no(text: str) -> str:
    t = normalise(text)
    match = re.search(r"\b(yes|no)\b", t, flags=re.I)
    if match:
        return match.group(1)
    return t


# ── Metrics ───────────────────────────────────────────────────────────────────

def token_f1(pred: str, ref: str) -> float:
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
    weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
    return sentence_bleu(
        [ref.lower().split()], pred.lower().split(), weights=weights,
        smoothing_function=SmoothingFunction().method1,
    )


def compute_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
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
        "yes_no_accuracy": yes_no_correct / yes_no_total if yes_no_total else 0.0,
        "yes_no_total": yes_no_total,
        "exact_match": avg(free_em),
        "macro_f1": avg(free_f1),
        "bleu1": avg(free_bleu1),
        "bleu2": avg(free_bleu2),
        "bleu3": avg(free_bleu3),
        "freeform_total": free_total,
        "total": yes_no_total + free_total,
    }


def preprocess_logits_for_metrics(logits, labels):
    return logits[0].argmax(dim=-1)


def make_compute_metrics(proc):
    def _compute(eval_pred):
        pred_ids, labels = eval_pred
        vocab_size = proc.tokenizer.vocab_size

        predictions, references = [], []
        for pred_row, label_row in zip(pred_ids, labels):
            mask = label_row != -100
            mask_indices = np.where(mask)[0]
            pred_indices = np.clip(mask_indices - 1, 0, len(pred_row) - 1)  # pred[j-1] predicts label[j]
            pred_tokens = np.clip(pred_row[pred_indices], 0, vocab_size - 1)
            label_tokens = label_row[mask_indices]
            predictions.append(proc.tokenizer.decode(pred_tokens, skip_special_tokens=True))
            references.append(proc.tokenizer.decode(label_tokens, skip_special_tokens=True))

        metrics = compute_metrics(predictions, references)
        metrics["combined_score"] = (metrics["bleu1"] + metrics["yes_no_accuracy"]) / 2
        return metrics
    return _compute