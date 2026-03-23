"""
Entity extraction and relation generation from pathology images using a VLM (Qwen2.5-VL).
Adapted from EntiGraph (Yang et al., 2025) — Steps 1 & 2: Entity Extraction + Relation Generation.

Step 1 – Entity extraction:
  For each image + caption pair, the model extracts a structured JSON with:
    - summary: brief description of the image
    - entities: list of salient visual/conceptual entities

Step 2 – Relation generation:
  For each pair of entities extracted in Step 1, the model
  generates free-text discussions of how those entities relate within the image/caption.

Results are saved incrementally to JSONL files so both steps are resumable.

Usage:
    # Step 1 — extract entities
    python extract_entities.py --images-dir output_images/ --mode entities

    # Step 2 — generate relations (reads entities JSONL produced by step 1)
    python extract_entities.py --images-dir output_images/ --mode relations [--triples]
"""

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import pandas as pd
import PIL.Image
import torch
sys.path.append(str(Path(__file__).parent / "src"))
from utils.loader import load_base_model

# ---------------------------------------------------------------------------
# Prompts
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

SYSTEM_PROMPT_EPIGRAPH = """
You will act as a knowledge analyzer tasked with dissecting an image-caption pair provided by the user. Your role involves two main objectives:

1. Rephrasing Content: The user will identify two specific entities mentioned in the pair. You are required to rephrase the content of the caption twice:
    * Once, emphasizing the first entity.
    * Again, emphasizing the second entity.
2. Analyzing Interactions: Discuss how the two specified entities interact within the context of the image-caption pair and how they relate to the image.

Your responses should provide clear segregation between the rephrased content and the interaction analysis. Ensure each section of the output include sufficient context, ideally referencing the image ID to maintain clarity about the discussion's focus. Here is the format you should follow for your response:
### Discussion of <image_id> in relation to <entity1>
<Rephrased content focusing on the first entity>

### Discussion of <image_id> in relation to <entity2>
<Rephrased content focusing on the second entity>

### Discussion of Interaction between <entity1> and <entity2> in context of <image_id>
<Discussion on how the two entities interact within the image>
"""

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def build_messages(img: PIL.Image.Image, user_prompt: str, system_prompt: str) -> list[dict]:
    """Build a chat message list for the VLM, with a configurable system prompt."""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def _run_inference(model, processor, messages: list[dict], img: PIL.Image.Image, max_new_tokens: int, do_sample: bool = False) -> str:
    """Tokenize, run the model, and return the decoded output string."""
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

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)

    return processor.batch_decode(
        generated_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def generate_and_parse(model, processor, img: PIL.Image.Image, caption: str, max_new_tokens: int, max_retries: int = 3) -> dict:
    """Generate entity extraction output (JSON), retrying up to max_retries times on parse failure."""
    user_prompt = f"### Caption:\n{caption}\n"
    messages = build_messages(img, user_prompt, system_prompt=SYSTEM_PROMPT_ENTITY_EXTRACTION)

    for attempt in range(1, max_retries + 1):
        raw = _run_inference(model, processor, messages, img, max_new_tokens, do_sample=attempt > 1)
        parsed = parse_json_output(raw)
        if "parse_error" not in parsed:
            return parsed
        print(f"    JSON parse failed (attempt {attempt}/{max_retries}), retrying...")

    print(f"    All {max_retries} attempts failed, saving raw output.")
    return parsed


def generate_relation(model, processor, img: PIL.Image.Image, caption: str, entities: list[str], max_new_tokens: int) -> str:
    """Generate a free-text relation discussion for 2 or 3 entities.

    Reuses build_messages and _run_inference; selects the right system prompt
    based on the number of entities passed.
    """
    assert 2 <= len(entities) <= 3, "Only 2- or 3-entity relations are supported."

    entity_lines = "\n".join(f"- {e}" for e in entities)
    user_prompt = f"### Caption:\n{caption}\n\n### Entities:\n{entity_lines}\n"

    messages = build_messages(img, user_prompt, system_prompt=SYSTEM_PROMPT_EPIGRAPH)
    return _run_inference(model, processor, messages, img, max_new_tokens)


def parse_json_output(raw: str) -> dict:
    """Extract JSON from model output, stripping any accidental markdown fences."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_output": raw, "parse_error": True}


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def load_already_processed(output_path: Path) -> set[str]:
    """Return set of image IDs already written to the entities JSONL."""
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


def load_already_processed_relations(output_path: Path) -> set[tuple]:
    """Return set of (image_id, entity_tuple) pairs already written to the relations JSONL."""
    if not output_path.exists():
        return set()
    done = set()
    with open(output_path) as f:
        for line in f:
            try:
                record = json.loads(line)
                done.add((record["id"], tuple(record["entities"])))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


# ---------------------------------------------------------------------------
# Step 1 — Entity extraction
# ---------------------------------------------------------------------------

def extract_entities(images_dir: Path, output_path: Path, model_name: str, cache_dir: str, quantize: bool, max_new_tokens: int, max_retries: int, start: int = 0, end: int = None):
    csv_path = images_dir / "images.csv"
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df["id"].apply(lambda x: (images_dir / f"{x}.jpg").exists())].reset_index(drop=True)
    df = df.iloc[start:end].reset_index(drop=True)
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
# Step 2 — Relation generation
# ---------------------------------------------------------------------------

def generate_relations(
    images_dir: Path,
    entities_path: Path,
    output_path: Path,
    model_name: str,
    cache_dir: str,
    quantize: bool,
    max_new_tokens: int,
    start: int = 0,
    end: int = None,
):
    """For each image, generate pairwise entity-relation discussions."""
    # Load entity records
    records = []
    with open(entities_path) as f:
        for line in f:
            records.append(json.loads(line))

    records = records[start:end]

    already_done = load_already_processed_relations(output_path)

    # Build the full work list upfront so we can report progress
    work = []
    for record in records:
        entities = record["entities"]
        pairs = list(itertools.combinations(entities, 2))
        for combo in pairs:
            key = (record["id"], combo)
            if key not in already_done:
                work.append((record, combo))

    print(f"{len(already_done)} relations already processed. {len(work)} remaining.")
    if not work:
        print("Nothing to do.")
        return

    print(f"Loading model {model_name} ...")
    model, processor = load_base_model(model_name, cache_dir=cache_dir, quantize=quantize)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors = 0

    with open(output_path, "a") as out_f:
        for i, (record, entity_combo) in enumerate(work):
            img_path = images_dir / f"{record['id']}.jpg"
            caption = record["caption"]

            try:
                img = PIL.Image.open(img_path).convert("RGB")
                relation_text = generate_relation(
                    model, processor, img, caption,
                    entities=list(entity_combo),
                    max_new_tokens=max_new_tokens,
                )
            except Exception as e:
                print(f"  [ERROR] {record['id']} {entity_combo}: {e}")
                relation_text = f"ERROR: {e}"
                errors += 1

            out_record = {
                "id": record["id"],
                "page": record.get("page"),
                "caption": caption,
                "entities": list(entity_combo),
                "relation": relation_text,
            }
            out_f.write(json.dumps(out_record) + "\n")
            out_f.flush()

            print(f"[{i+1}/{len(work)}] {record['id']} — {entity_combo}")

    print(f"\nDone. Relations written to {output_path}. Errors: {errors}/{len(work)}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="EntiGraph pipeline for pathology images: entity extraction + relation generation.")
    parser.add_argument("--images-dir", type=Path, default=Path("output_images"),
                        help="Directory with images and images.csv (default: output_images/)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B",
                        help="HuggingFace model name (default: Qwen/Qwen3.5-9B)")
    parser.add_argument("--cache-dir", type=str, default="/Data/joao.giordani-donasolo/huggingface",
                        help="HuggingFace cache dir")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Disable 4-bit NF4 quantization")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max tokens to generate per image (default: 512)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max JSON parse retries per image (default: 3)")
    parser.add_argument("--start", type=int, default=0,
                        help="First image index to process, inclusive (default: 0)")
    parser.add_argument("--end", type=int, default=None,
                        help="Last image index to process, exclusive (default: all)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    entities_path = args.images_dir / "entities.jsonl"
    relations_path = args.images_dir / "relations.jsonl"

    extract_entities(
        images_dir=args.images_dir,
        output_path=entities_path,
        model_name=args.model,
        cache_dir=args.cache_dir,
        quantize=not args.no_quantize,
        max_new_tokens=args.max_new_tokens,
        max_retries=args.max_retries,
        start=args.start,
        end=args.end,
    )
    generate_relations(
        images_dir=args.images_dir,
        entities_path=entities_path,
        output_path=relations_path,
        model_name=args.model,
        cache_dir=args.cache_dir,
        quantize=not args.no_quantize,
        max_new_tokens=args.max_new_tokens,
        start=args.start,
        end=args.end,
    )