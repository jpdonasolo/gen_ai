import PIL

import os
import base64
from io import BytesIO

import torch

import datasets
from trl import SFTTrainer, SFTConfig

from utils import load_base_model, load_dataset


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


def main():
    global processor

    ds_qa = load_dataset(cache_dir="huggingface/")
    model, processor = load_base_model("Qwen/Qwen3.5-0.8B-Base", peft=True, peft_config={"r": 32})
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
        eval_steps=30,
        eval_strategy="steps",
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
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
        args=config
    )
    trainer.train()
    trainer.save_model("results/baseline")

if __name__ == "__main__":
    print(f"Using device {DEVICE}")
    main()