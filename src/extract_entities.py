"""
Entity extraction from pathology images using a VLM (Qwen3.5-9B).
Adapted from EntiGraph (Yang et al., 2025) — Step 1: Entity Extraction.

For each image + caption pair, the model extracts a structured JSON with:
  - summary: brief description of the image
  - entities: list of salient visual/conceptual entities

Results are saved incrementally to a JSONL file so the script is resumable.

Usage:
    python extract_entities.py --images-dir output_images/ [--output output_images/entities.jsonl]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import PIL.Image
import torch
import sys
sys.path.append(str(Path(__file__).parent / "src"))
from utils.loader import load_base_model

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_ENTITY_EXTRACTION = """
As a knowledge analyzer, your task is to dissect and understand an image-caption pair provided by the user. You are required to perform the following steps:
1. Summarize: Provide a concise summary of the entire image-caption, capturing the main points and themes.
2. Extract Entities: Identify and list all significant "nouns" or entities relevant to the image. These entities should include but not limited to:
* Objects: All objects present, including organs, human tissues, cells and annotation such as arrows.
* Pathological findings: Such as necrosis, fibrosis, hyperplasia.
* Staining methods: H&E, PAS, immunohistochemistry.
* Diseases: Diseases or conditions depicted or referenced.
Try to exhaust as many entities as possible. Your response should be structured in a JSON format to organize the information effectively.
Ensure that the summary is brief yet comprehensive, and the list of entities is detailed and accurate.
Here is the format you should use for your response:
{
"summary": "<A concise summary of the article>",
"entities": ["entity1", "entity2", ...]
}
"""

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def build_messages(img: PIL.Image.Image, caption: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT_ENTITY_EXTRACTION}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": f"### Caption:\n{caption}\n"},
            ],
        },
    ]


def generate_and_parse(model, processor, img: PIL.Image.Image, caption: str, max_new_tokens: int, max_retries: int = 3) -> dict:
    """Generate entity extraction output, retrying up to max_retries times on JSON parse failure."""
    messages = build_messages(img, caption)
    text = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    device = next(model.parameters()).device
    inputs = processor(text=[text], images=[img], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    for attempt in range(1, max_retries + 1):
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=attempt > 1)
        raw = processor.batch_decode(
            generated_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        parsed = parse_json_output(raw)
        if "parse_error" not in parsed:
            return parsed
        print(f"    JSON parse failed (attempt {attempt}/{max_retries}), retrying...")

    print(f"    All {max_retries} attempts failed, saving raw output.")
    return parsed


def parse_json_output(raw: str) -> dict:
    """Extract JSON from model output, stripping any accidental markdown fences."""
    raw = raw.strip()
    # Strip ```json ... ``` if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Return raw output so the record is still saved
        return {"raw_output": raw, "parse_error": True}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def load_already_processed(output_path: Path) -> set[str]:
    """Return set of image IDs already written to the output JSONL."""
    if not output_path.exists():
        return set()
    done = set()
    with open(output_path) as f:
        for line in f:
            try:
                record = json.loads(line)
                done.add(record["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def run(images_dir: Path, output_path: Path, model_name: str, cache_dir: str, quantize: bool, max_new_tokens: int, max_retries: int):
    csv_path = images_dir / "images.csv"
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Keep only rows whose image file actually exists
    df = df[df["id"].apply(lambda x: (images_dir / f"{x}.jpg").exists())].reset_index(drop=True)
    print(f"Found {len(df)} images with captions.")

    already_done = load_already_processed(output_path)
    df = df[~df["id"].isin(already_done)].reset_index(drop=True)
    print(f"{len(already_done)} already processed. {len(df)} remaining.")

    if df.empty:
        print("Nothing to do.")
        return

    print(f"Loading model {model_name} ...")
    model, processor = load_base_model(model_name, cache_dir=cache_dir, quantize=quantize)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors = 0

    with open(output_path, "a") as out_f:
        for i, row in df.iterrows():
            img_path = images_dir / f"{row['id']}.jpg"
            caption = str(row.get("caption", "")).strip()

            try:
                img = PIL.Image.open(img_path).convert("RGB")
                parsed = generate_and_parse(model, processor, img, caption, max_new_tokens=max_new_tokens, max_retries=max_retries)
            except Exception as e:
                print(f"  [ERROR] {row['id']}: {e}")
                parsed = {"error": str(e)}
                errors += 1

            record = {
                "id": row["id"],
                "page": row.get("page"),
                "caption": caption,
                **parsed,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            n_entities = len(parsed.get("entities", []))
            print(f"[{i+1}/{len(df)+len(already_done)}] {row['id']} — {n_entities} entities")

    print(f"\nDone. Results written to {output_path}. Errors: {errors}/{len(df)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Extract visual entities from pathology images.")
    parser.add_argument("--images-dir", type=Path, default=Path("output_images"),
                        help="Directory with images and images.csv (default: output_images/)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSONL file (default: <images-dir>/entities.jsonl)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B",
                        help="HuggingFace model name (default: Qwen/Qwen3.5-9B)")
    parser.add_argument("--cache-dir", type=str, default="huggingface",
                        help="HuggingFace cache dir (default: huggingface/)")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Disable 4-bit NF4 quantization")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max tokens to generate per image (default: 512)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max JSON parse retries per image (default: 3)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_path = args.output or (args.images_dir / "entities.jsonl")
    run(
        images_dir=args.images_dir,
        output_path=output_path,
        model_name=args.model,
        cache_dir=args.cache_dir,
        quantize=not args.no_quantize,
        max_new_tokens=args.max_new_tokens,
        max_retries=args.max_retries,
    )