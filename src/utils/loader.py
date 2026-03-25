import PIL
import json
import os
from pathlib import Path

import pandas as pd
import torch
import datasets
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model, PeftModel

from .config import DEVICE


###############
###  Model  ###
###############

def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config(config: dict = None):
    if config is None:
        config = {}
    return LoraConfig(
        r=config.get("r", 64),
        lora_alpha=config.get("lora_alpha", 64),
        lora_dropout=config.get("lora_dropout", 0.2),
        target_modules=config.get("target_modules", "all-linear"),
        task_type=TaskType.CAUSAL_LM,
        bias=config.get("bias", "none"),
    )


def load_base_model(
    model_name: str,
    cache_dir: str = "huggingface",
    quantize: bool = True,
    peft: bool = False,
    peft_config: dict = None,
):
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    processor.tokenizer.padding_side = "left"

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=get_bnb_config() if quantize else None,
    ).to(DEVICE)

    if peft:
        model = prepare_model_for_kbit_training(model)
        config = get_lora_config(peft_config)
        model = get_peft_model(model, peft_config=config)

    return model, processor


def load_lora_pretrained_model(checkpoint: str, *args, is_trainable: bool = False, **kwargs):
    if kwargs.get("peft", False):
        raise ValueError
    model, processor = load_base_model(*args, **kwargs)
    model = PeftModel.from_pretrained(model, checkpoint, is_trainable=is_trainable)
    return model, processor


###############
## Datasets  ##
###############

def add_prefix(obs):
    question: str = obs["question"]
    if question.startswith("does") or question.startswith("is"):
        question = "answer strictly with yes or no. " + question
    else:
        question = "answer with as few words as possible. " + question
    obs["question"] = question
    return obs


def load_base_dataset(add_prefixes: bool = False, *args, **kwargs):
    ds = datasets.load_dataset("flaviagiammarino/path-vqa", *args, **kwargs)
    if add_prefixes:
        ds = ds.map(add_prefix, num_proc=os.cpu_count())
    return ds


def load_epigraph(cache_dir: str = "huggingface", k: int = 20) -> datasets.Dataset:
    if k >= 200:
        processed_path = os.path.join(cache_dir, f"relations_dataset")
    else:
        processed_path = os.path.join(cache_dir, f"relations_{k}")
    
    if os.path.exists(processed_path):
        return datasets.load_from_disk(processed_path).rename_column("relation", "text")
        
    relations = []
    for j in Path("output_images/").glob("relations_*.jsonl"):
        with open(j, "r") as f:
            lines = f.readlines()
        for l in lines:
            relations.append(json.loads(l))

    df = pd.DataFrame(relations)
    df_sampled = df.groupby("id").apply(
        lambda grp: grp.sample(
            n=min(k, len(grp)),
            random_state=0,
            replace=False,
        )
    )
    df_sampled.index = df_sampled.index.get_level_values("id")
    df_sampled = df_sampled.reset_index(drop=False)

    df_sampled["image"] = df_sampled["id"].apply(lambda x: f"output_images/{x}.jpg")

    ds = datasets.Dataset.from_pandas(df_sampled)
    ds = ds.cast_column("image", datasets.Image())
    ds.save_to_disk(processed_path)
    
    return ds.rename_column("relation", "text")


def load_redpajama(tokenizer=None, max_length: int = None, cache_dir: str = "huggingface") -> datasets.Dataset:
    processed_path = os.path.join(cache_dir, f"redpajama_truncated_{max_length}")

    if tokenizer is not None and max_length is not None:
        if os.path.exists(processed_path):
            return datasets.load_from_disk(processed_path)

    ds = datasets.load_dataset(
        "krisbailey/RedPajama-1B-Weighted",
        split="train",
        cache_dir=cache_dir,
    )

    if tokenizer is not None and max_length is not None:
        def truncate_text(example):
            tokens = tokenizer(example["text"], truncation=True, max_length=max_length, add_special_tokens=False)
            example["text"] = tokenizer.decode(tokens["input_ids"])
            return example

        ds = ds.map(truncate_text, num_proc=os.cpu_count())
        ds.save_to_disk(processed_path)

    return ds