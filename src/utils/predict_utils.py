import re
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


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
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
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
    """Return 'yes' or 'no' if present in the prediction, else the raw text."""
    t = normalise(text)
    if "yes" in t.split():
        return "yes"
    if "no" in t.split():
        return "no"
    return t


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    predictions: list[str], references: list[str]
) -> dict[str, float]:
    """
    Returns yes_no_acc, freeform_acc, overall_acc.
    All inputs should already be lowercased strings.
    """
    yes_no_correct, yes_no_total = 0, 0
    free_correct, free_total = 0, 0

    for pred, ref in zip(predictions, references):
        ref_norm = normalise(ref)
        if is_yes_no(ref_norm):
            yes_no_total += 1
            yes_no_correct += int(extract_yes_no(pred) == ref_norm)
        else:
            free_total += 1
            free_correct += int(normalise(pred) == ref_norm)

    total = yes_no_total + free_total
    return {
        "yes_no_accuracy": yes_no_correct / yes_no_total if yes_no_total else 0.0,
        "freeform_accuracy": free_correct / free_total if free_total else 0.0,
        "overall_accuracy": (yes_no_correct + free_correct) / total if total else 0.0,
        "yes_no_total": yes_no_total,
        "freeform_total": free_total,
        "total": total,
    }