import os
import torch
import argparse

from trl import SFTTrainer, SFTConfig

from utils import load_base_model, load_epigraph, load_redpajama, MixedDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR = "huggingface"
MAX_LEN = 256
processor = None  # Global used in collate_fn


def preprocess_epigraph(example):
    img = example["image"]
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": example["text"]},
        ]
    }]
    text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"prompt": text, "images": [img]}

def preprocess_redpajama(example):
    return {"prompt": example["text"], "images": []}

def collate_fn(examples):
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--experiment-name", type=str, required=True)
    return parser.parse_args()


def main(args):
    global processor

    model, processor = load_base_model(args.model, cache_dir=CACHE_DIR, peft=True, peft_config={"r": 32})
    model.print_trainable_parameters()

    epigraph_ds = load_epigraph(cache_dir=CACHE_DIR)
    epigraph_ds = epigraph_ds.map(
        preprocess_epigraph, remove_columns=epigraph_ds.column_names, num_proc=os.cpu_count()
    )

    redpajama_ds = load_redpajama(tokenizer=processor.tokenizer, max_length=MAX_LEN, cache_dir=CACHE_DIR)
    redpajama_ds = redpajama_ds.map(
        preprocess_redpajama, remove_columns=redpajama_ds.column_names, num_proc=os.cpu_count()
    )

    train_ds = MixedDataset(epigraph_ds, redpajama_ds)

    if args.max_train_samples is not None:
        from torch.utils.data import Subset
        n = min(args.max_train_samples, len(train_ds))
        train_ds = Subset(train_ds, range(n))
        print(f"Using {n} batches")

    config = SFTConfig(
        output_dir=f"results/{args.experiment_name}",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=32,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=1e-5,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=1,
        save_steps=20,
        save_total_limit=10,
        report_to="wandb",
        run_name=args.experiment_name,
        max_length=MAX_LEN,
        seed=42,
        gradient_checkpointing=True
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        train_dataset=train_ds,
        data_collator=collate_fn,
        args=config,
    )
    trainer.train()
    trainer.save_model(f"results/{args.experiment_name}/final")


if __name__ == "__main__":
    print(f"Using device {DEVICE}")
    args = parse_args()
    main(args)