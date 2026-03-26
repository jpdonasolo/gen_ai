import os

LRS = ["1.0e-5", "2.0e-5", "5.0e-5"]
BATCH_SIZES = [16, 32, 64]
LORA_RS = [16, 32, 64]
MERGE_VQAS = [True, False]

TEMPLATE = """
experiment_name: "{merge_vqa_flag}_lora{lora}_lr{lr}_bs{batch_size}"
model: "Qwen/Qwen3.5-0.8B-Base"
cache_dir: "huggingface"
max_train_samples: null

replay:
  epigraph_k: 200
  rp_max_len: 256
  merge_with_vqa: {merge_vqa}

lora:
  r: {lora}
  lora_alpha: {lora}

training:
  num_train_epochs: 3
  learning_rate: {lr}
  lr_scheduler_type: "cosine"
  warmup_steps: 50
  weight_decay: 5.0e-6

  per_device_train_batch_size: 2
  gradient_accumulation_steps: {batch_size}
  gradient_checkpointing: true
  report_to: "wandb"
"""

def main():
    os.makedirs("configs", exist_ok=True)
    for merge_vqa in MERGE_VQAS:
        for lora in LORA_RS:
            for lr in LRS:
                for batch_size in BATCH_SIZES:
                    merge_vqa_str = "withvqa" if merge_vqa else "novqa"
                    exp_name = f"{merge_vqa_str}_lora{lora}_lr{lr}_bs{batch_size}"
                    config_str = TEMPLATE.format(
                        merge_vqa=merge_vqa,
                        merge_vqa_flag=merge_vqa_str,
                        lora=lora,
                        lr=lr,
                        batch_size=batch_size
                    )
                    path = f"configs/{exp_name}.yaml"
                    with open(path, "w") as f:
                        f.write(config_str)
                    print(f"Wrote {path}")

if __name__ == "__main__":
    main()