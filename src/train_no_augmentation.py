import os
import argparse
from trl import SFTTrainer

from utils import (
    load_base_model, load_base_dataset,
    make_compute_metrics, preprocess_logits_for_metrics,
    preprocess_vqa, make_collate_vqa,
)
from utils.config import DEVICE, CACHE_DIR, base_sft_config, add_common_train_args


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_train_args(parser)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--add-prefix", action="store_true", help="Add instruction prefix to dataset.")
    return parser.parse_args()


def main(args):
    ds_qa = load_base_dataset(args.add_prefix, cache_dir=CACHE_DIR)
    model, processor = load_base_model(args.model, peft=True)
    model.print_trainable_parameters()

    collate_vqa = make_collate_vqa(processor)

    ds_qa_train = ds_qa["train"].map(
        preprocess_vqa,
        remove_columns=ds_qa["train"].column_names,
        num_proc=os.cpu_count(),
        writer_batch_size=500,
    )
    ds_qa_val = ds_qa["validation"].map(
        preprocess_vqa,
        remove_columns=ds_qa["validation"].column_names,
        num_proc=os.cpu_count(),
        writer_batch_size=500,
    )

    if args.max_train_samples is not None:
        ds_qa_train = ds_qa_train.select(range(min(args.max_train_samples, len(ds_qa_train))))
        print(f"Using {len(ds_qa_train)} train samples")
    if args.max_eval_samples is not None:
        ds_qa_val = ds_qa_val.select(range(min(args.max_eval_samples, len(ds_qa_val))))
        print(f"Using {len(ds_qa_val)} val samples")

    config = base_sft_config(
        args.experiment_name,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=3,
        fp16=False,
        eval_steps=20,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="combined_score",
        greater_is_better=True,
        max_length=512,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        train_dataset=ds_qa_train,
        eval_dataset=ds_qa_val,
        data_collator=collate_vqa,
        compute_metrics=make_compute_metrics(processor),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        args=config,
    )
    trainer.train()
    trainer.save_model(f"results/{args.experiment_name}/final")


if __name__ == "__main__":
    print(f"Using device {DEVICE}")
    args = parse_args()
    main(args)