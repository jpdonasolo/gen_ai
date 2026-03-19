import torch

from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    processor.tokenizer.padding_side = "left"
    
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=get_bnb_config() if quantize else None,
    ).to(DEVICE)

    if peft:
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config=get_lora_config())
    
    return model, processor
