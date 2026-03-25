import os
import argparse
from trl import SFTTrainer

from utils import (
    load_base_model, load_base_dataset,
    preprocess_vqa, make_collate,
    load_lora_pretrained_model
)
from utils.config import DEVICE, CACHE_DIR, base_sft_config, add_common_train_args


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_train_args(parser)
    # parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--add-prefix", action="store_true", help="Add instruction prefix to dataset.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cache-dir", default="huggingface")
    return parser.parse_args()


def main(args):
    ds_qa = load_base_dataset(args.add_prefix, cache_dir=CACHE_DIR)

    if args.checkpoint is None:
        print(f"Loading model: {args.model}")
        model, processor = load_base_model(args.model, peft=True)
    else:
        print(f"Loading pretrained model: {args.checkpoint} from {args.model}")
        model, processor = load_lora_pretrained_model(args.checkpoint, True, args.model, args.cache_dir)
    
    model.print_trainable_parameters()

    collate_fn = make_collate(processor, mask_prompt=True)

    ds_qa_train = ds_qa["train"].map(
        preprocess_vqa,
        remove_columns=ds_qa["train"].column_names,
        num_proc=os.cpu_count(),
        writer_batch_size=500,
    )

    if args.max_train_samples is not None:
        ds_qa_train = ds_qa_train.select(range(min(args.max_train_samples, len(ds_qa_train))))
        print(f"Using {len(ds_qa_train)} train samples")

    config = base_sft_config(
        args.experiment_name,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=3,
        fp16=False,
        max_length=512,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        train_dataset=ds_qa_train,
        data_collator=collate_fn,
        args=config,
    )
    trainer.train()
    trainer.save_model(f"results/{args.experiment_name}/final")


if __name__ == "__main__":
    print(f"Using device {DEVICE}")
    args = parse_args()
    main(args)