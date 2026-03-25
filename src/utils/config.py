import torch
from trl import SFTConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR = "huggingface"


def add_common_train_args(parser):
    """Args shared by every training script."""
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--experiment-name", type=str, required=True)
    return parser


def base_sft_config(experiment_name: str, **overrides) -> SFTConfig:
    """SFTConfig with shared defaults. Pass per-script overrides as kwargs."""
    defaults = dict(
        num_train_epochs=1,
        gradient_accumulation_steps=32,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=1e-5,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=1,
        save_steps=20,
        save_total_limit=10,
        # report_to="wandb",
        seed=42,
    )
    defaults.update(overrides)
    return SFTConfig(
        output_dir=f"results/{experiment_name}",
        run_name=experiment_name,
        **defaults,
    )