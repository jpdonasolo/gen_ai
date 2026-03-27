from argparse import ArgumentParser
from pathlib import Path

from torch.utils.data import Subset
from trl import SFTTrainer

from utils import load_base_model, get_replay_dataset, make_collate
from utils.config import DEVICE, base_sft_config
from utils.train_yaml import load_train_yaml


def parse_args():
    parser = ArgumentParser(description="Replay-style training.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help=f"YAML config path",
    )
    return parser.parse_args()


def main(args):
    cfg = load_train_yaml(args.config)

    tr = cfg["training"]
    lora_cfg = cfg["lora"]
    replay_cfg = cfg["replay"]
    pd_bs = tr["per_device_train_batch_size"]
    gas = tr["gradient_accumulation_steps"]
    print(
        "\n=== Run configuration ===\n"
        f"  Base model:           {cfg['model']}\n"
        f"  LoRA r:               {lora_cfg['r']}\n"
        f"  Batch (device x GA):  {pd_bs} x {gas} (effective {pd_bs * gas} per step, 1 GPU)\n"
        f"  Merge with VQA:       {replay_cfg['merge_with_vqa']}\n"
        f"  Learning rate:        {tr['learning_rate']}\n"
        f"  Train epochs:         {tr['num_train_epochs']}\n"
    )

    cache_dir = cfg.get("cache_dir") or "huggingface"
    replay = cfg.get("replay") or {}
    lora = cfg.get("lora") or {}
    training = dict(cfg.get("training") or {})

    rp_max_len = replay.get("rp_max_len", 256)
    training.setdefault("max_length", rp_max_len)

    peft_config = {
        "r": lora.get("r", 64),
        "lora_alpha": lora.get("lora_alpha", 64),
    }
    model, processor = load_base_model(
        cfg["model"], cache_dir=cache_dir, peft=True, peft_config=peft_config
    )
    model.print_trainable_parameters()

    collate_fn = make_collate(processor, mask_prompt=True)
    train_ds = get_replay_dataset(
        processor, 
        cache_dir, 
        rp_max_len, 
        epigraph_k=replay.get("epigraph_k", 20),
        merge_with_vqa=replay.get("merge_with_vqa", True),
        use_aux_ds=False,
        instruct=True,
    )

    max_train = cfg.get("max_train_samples")
    if max_train is not None:
        n = min(max_train, len(train_ds))
        train_ds = Subset(train_ds, range(n))
        print(f"Using {n} batches")

    config = base_sft_config(cfg["experiment_name"], **training)

    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        train_dataset=train_ds,
        data_collator=collate_fn,
        args=config,
    )
    trainer.train()
    trainer.save_model(f"results/{cfg['experiment_name']}/final")


if __name__ == "__main__":
    print(f"Using device {DEVICE}")
    main(parse_args())
