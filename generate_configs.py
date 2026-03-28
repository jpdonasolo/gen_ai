import os

CONFIGS_DIR = "configs/instruct"

LRS = ["5.0e-6", "1.0e-5"]
BATCH_SIZES = [128, 256]
LORA_RS = [16, 32]
MERGE_VQAS = [True, False]

TEMPLATE = """
experiment_name: "instruct_{merge_vqa_flag}_lora{lora}_lr{lr}_bs{batch_size}_full"
model: "Qwen/Qwen3.5-0.8B"
cache_dir: "huggingface"
max_train_samples: null

replay:
  epigraph_k: 200
  rp_max_len: {max_length}
  merge_with_vqa: {merge_vqa}
  load_epigraph_full: true

lora:
  r: {lora}
  lora_alpha: {lora}

training:
  num_train_epochs: 3
  learning_rate: {lr}
  lr_scheduler_type: "cosine"
  warmup_steps: 50
  weight_decay: 5.0e-6
  
  save_steps: 40
  save_total_limit: 100

  per_device_train_batch_size: 1
  gradient_accumulation_steps: {batch_size}
  gradient_checkpointing: true
  report_to: "wandb"
"""

def main():
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    for merge_vqa in MERGE_VQAS:
        for lora in LORA_RS:
            for lr in LRS:
                for batch_size in BATCH_SIZES:
                    merge_vqa_str = "withvqa" if merge_vqa else "novqa"
                    exp_name = f"instruct_{merge_vqa_str}_lora{lora}_lr{lr}_bs{batch_size}_full"
                    config_str = TEMPLATE.format(
                        merge_vqa=merge_vqa,
                        merge_vqa_flag=merge_vqa_str,
                        lora=lora,
                        lr=lr,
                        batch_size=batch_size,
                        max_length=256
                    )
                    path = os.path.join(CONFIGS_DIR, f"{exp_name}.yaml")
                    with open(path, "w") as f:
                        f.write(config_str)
                    print(f"Wrote {path}")

if __name__ == "__main__":
    main()