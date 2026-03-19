# %%
import PIL

import os
import base64
from io import BytesIO

import torch

import datasets
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# %%
ds_qa = datasets.load_dataset("flaviagiammarino/path-vqa", cache_dir="huggingface/")
# ds_pm = datasets.load_dataset("FreedomIntelligence/PubMedVision", "PubMedVision_Alignment_VQA", cache_dir="huggingface/")

# %%
def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

def get_lora_config():
    """Build LoRA config from the config dict."""
    return LoraConfig(
        r=64,
        lora_alpha=64,
        lora_dropout=0.2,
        # target_modules=[
        #     "q_proj", "w_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        #     "proj", "qkv", "linear_fc1", "linear_fc2"
        # ],
        target_modules="all-linear", 
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

def load_model(
    model_name: str, 
    cache_dir: str = "huggingface",
    quantize: bool = True,
    peft: bool = False
):  
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name, 
        cache_dir=cache_dir,
        quantization_config=get_bnb_config() if quantize else None
    ).to(DEVICE)
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)

    if quantize:
        model = prepare_model_for_kbit_training(model)

    if peft:
        model = get_peft_model(model, peft_config=get_lora_config())

    return model, processor


model, processor = load_model("Qwen/Qwen3.5-0.8B", peft=True)

# %%
model.print_trainable_parameters()

# %%
processor.tokenizer.padding_side

# %%
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

# %%
def collate_fn(examples):
    texts, images = [], []

    for example in examples:
        messages = example["prompt"] + example["completion"]  # full conversation
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
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

# %%
config = SFTConfig(
    output_dir="results/baseline",
    num_train_epochs=3,
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
    save_steps=60,
    eval_steps=30,
    eval_strategy="steps",
    save_total_limit=2,
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