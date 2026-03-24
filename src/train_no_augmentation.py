import PIL

import os
import base64
from io import BytesIO

import torch

import datasets
from trl import SFTTrainer, SFTConfig

import argparse
import numpy as np
from utils import load_base_model, load_dataset, compute_metrics


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = None # Use as global variable in collation function


def preprocess_dataset(example):
    
    # buf = BytesIO()
    img = example["image"]
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    # img.save(buf, format="JPEG", quality=85)  # JPEG >> PNG speed
    # img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    return {
        "prompt": [{"role": "user", "content": [
            {"type": "text", "text": example["question"]},
            {"type": "image"},
        ]}],
        "completion": [{"role": "assistant", "content": example["answer"]}],
        "images": [example["image"]]
    }

def collate_fn(examples):
    texts, images = [], []

    for example in examples:
        messages = example["prompt"] + example["completion"]  # full conversation
        text = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False
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


def preprocess_logits_for_metrics(logits, labels):
    return logits[0].argmax(dim=-1)

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    return parser.parse_args()

def main(args):
    global processor

    ds_qa = load_dataset(cache_dir="huggingface/")
    model, processor = load_base_model(args.model, peft=True, peft_config={"r": 32})
    model.print_trainable_parameters()

    ds_qa_train = ds_qa["train"].map(
        preprocess_dataset,
        remove_columns=ds_qa["train"].column_names,
        num_proc=os.cpu_count(),
        writer_batch_size=500,
    )
    ds_qa_val = ds_qa["validation"].map(
        preprocess_dataset,
        remove_columns=ds_qa["validation"].column_names,
        num_proc=os.cpu_count(),
        writer_batch_size=500,
    )

    if args.max_train_samples is not None:
        ds_qa_train = ds_qa_train.select(range(min(args.max_train_samples, len(ds_qa_train))))
        print(f"Using {len(ds_qa_train)} train samples")
    if args.max_eval_samples is not None:
        ds_qa_val = ds_qa_val.select(range(min(args.max_eval_samples, len(ds_qa_val))))
        print(f"Using {len(ds_qa_val)} val samples")

    config = SFTConfig(
        output_dir="results/baseline",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=3,
        gradient_accumulation_steps=32,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=1e-5,
        max_grad_norm=1.0,
        fp16=False,
        bf16=True,
        logging_steps=1,
        save_steps=30,
        eval_steps=1,
        eval_strategy="steps",
        save_total_limit=10,
        load_best_model_at_end=True,
        metric_for_best_model="combined_score",
        greater_is_better=True,
        report_to="wandb",
        run_name="multigen-baseline",
        max_length=512,
        dataset_text_field="text",
        seed=42,
    )


    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        train_dataset=ds_qa_train,
        eval_dataset=ds_qa_val,
        data_collator=collate_fn,
        compute_metrics=make_compute_metrics(processor),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        args=config
    )
    trainer.train()
    trainer.save_model("results/baseline/final")

if __name__ == "__main__":
    print(f"Using device {DEVICE}")
    args = parse_args()
    main(args)