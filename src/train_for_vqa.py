import os
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

from trl import SFTTrainer

from utils import (
    load_base_model,
    load_base_dataset,
    preprocess_vqa,
    make_collate,
    load_lora_pretrained_model,
)
from utils.config import DEVICE, base_sft_config
from utils.train_yaml import load_train_yaml


def parse_args():
    parser = ArgumentParser(description="Fine-tune on Path-VQA.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help=f"YAML config path",
    )
    parser.add_argument("--report-to", type=str, default=None, help="Override training.report_to.")
    return parser.parse_args()


def main(args):
    cfg = load_train_yaml(args.config)
    if hasattr(args, "report_to"):
       cfg["training"]["report_to"] = args.report_to

    cache_dir = cfg.get("cache_dir") or "huggingface"
    vqa = cfg.get("vqa") or {}
    training = dict(cfg.get("training") or {})

    ds_qa = load_base_dataset(vqa.get("add_prefix", False), cache_dir=cache_dir)

    checkpoint = vqa.get("checkpoint")
    if checkpoint is None:
        print(f"Loading model: {cfg['model']}")
        model, processor = load_base_model(cfg["model"], peft=True)
    else:
        print(f"Loading pretrained model: {checkpoint} from {cfg['model']}")
        model, processor = load_lora_pretrained_model(
            checkpoint, cfg["model"], cache_dir=cache_dir, is_trainable=True
        )

    model.print_trainable_parameters()

    collate_fn = make_collate(processor, mask_prompt=True)

    ds_qa_train = ds_qa["train"].map(
        preprocess_vqa,
        remove_columns=ds_qa["train"].column_names,
        num_proc=os.cpu_count(),
        writer_batch_size=500,
    )

    max_train = cfg.get("max_train_samples")
    if max_train is not None:
        ds_qa_train = ds_qa_train.select(range(min(max_train, len(ds_qa_train))))
        print(f"Using {len(ds_qa_train)} train samples")

    config = base_sft_config(cfg["experiment_name"], **training)

    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        train_dataset=ds_qa_train,
        data_collator=collate_fn,
        args=config,
    )
    trainer.train()
    trainer.save_model(f"results/{cfg['experiment_name']}/final")


if __name__ == "__main__":
    print(f"Using device {DEVICE}")
    main(parse_args())
