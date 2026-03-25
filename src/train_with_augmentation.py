import argparse
from torch.utils.data import Subset
from trl import SFTTrainer

from utils import load_base_model, get_replay_dataset, make_collate
from utils.config import DEVICE, CACHE_DIR, base_sft_config, add_common_train_args

MAX_LEN = 256


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_train_args(parser)
    return parser.parse_args()


def main(args):
    model, processor = load_base_model(args.model, cache_dir=CACHE_DIR, peft=True, peft_config={"r": 32})
    model.print_trainable_parameters()

    collate_fn = make_collate(processor, mask_prompt=False)
    train_ds = get_replay_dataset(processor, CACHE_DIR, MAX_LEN)

    if args.max_train_samples is not None:
        n = min(args.max_train_samples, len(train_ds))
        train_ds = Subset(train_ds, range(n))
        print(f"Using {n} batches")

    config = base_sft_config(
        args.experiment_name,
        per_device_train_batch_size=2,
        max_length=MAX_LEN,
        gradient_checkpointing=True,
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