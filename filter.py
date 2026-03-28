import os
import re
import hashlib
import argparse
import pandas as pd
from PIL import Image


COURTESY_PATTERN = re.compile(r"\(Courtesy.*?\)", re.IGNORECASE)
FIG_PATTERN = re.compile(r"^Fig(?:ure)?\.?\s*\d+[\.\d]*\.?\s*", re.IGNORECASE)
KEY_PATTERNS = re.compile(r"(?:^(?:Mechanism|Diagrammatic|Pathophysiology|Arachidonic|Sequence|Cellular)|[Ss]chematic)")

def hash_image(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


max_size = float("-inf")

def is_too_small(path, min_w=90, min_h=75):
    with Image.open(path) as img:
        w, h = img.size
    return w < min_w or h < min_h

def is_too_big(path, max_size=810_000):
    with Image.open(path) as img:
        w, h = img.size
    return w *h > max_size


def has_multiple_subfigures(caption):
    """Detect captions that have both (A) and (B) sub-figure labels."""
    if not isinstance(caption, str):
        return False
    return "(A)" in caption and "(B)" in caption

def filter_dataset(dataset_dir, remove_dir, min_w=90, min_h=75, key_patterns=False):
    remove_hashes = {
        hash_image(os.path.join(remove_dir, f))
        for f in os.listdir(remove_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    }

    duplicates, too_small, too_big = set(), set(), set()
    for fname in os.listdir(dataset_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        fpath = os.path.join(dataset_dir, fname)
        stem = os.path.splitext(fname)[0]

        if hash_image(fpath) in remove_hashes:
            os.remove(fpath)
            duplicates.add(stem)
        elif is_too_small(fpath, min_w, min_h):
            os.remove(fpath)
            too_small.add(stem)
        elif is_too_big(fpath):
            os.remove(fpath)
            too_big.add(stem)

    csv_path = os.path.join(dataset_dir, "images.csv")
    df = pd.read_csv(csv_path)

    n_duplicates = df["id"].isin(duplicates).sum()
    n_too_small = df["id"].isin(too_small).sum()
    n_too_big = df["id"].isin(too_big).sum()
    n_no_caption = (df["caption"].isna() | (df["caption"].astype(str).str.strip() == "")).sum()

    multi_subfig_mask = df["caption"].apply(has_multiple_subfigures)
    n_multi_subfig = multi_subfig_mask.sum()

    no_caption_stems = df[df["caption"].isna() | (df["caption"].astype(str).str.strip() == "")]["id"].tolist()
    multi_subfig_stems = df[multi_subfig_mask]["id"].tolist()

    for stem in no_caption_stems + multi_subfig_stems:
        for ext in (".jpg", ".jpeg", ".png"):
            fpath = os.path.join(dataset_dir, stem + ext)
            if os.path.exists(fpath):
                os.remove(fpath)

    df = df[~df["id"].isin(duplicates)]
    df = df[~df["id"].isin(too_small)]
    df = df[df["caption"].notna() & (df["caption"].astype(str).str.strip() != "")]
    df = df[~df["caption"].apply(has_multiple_subfigures)]

    # Clean courtesy attributions and leading figure labels from captions
    n_courtesy = df["caption"].str.contains(COURTESY_PATTERN, na=False).sum()
    n_fig_prefix = df["caption"].str.contains(FIG_PATTERN, na=False).sum()
    df["caption"] = df["caption"].str.replace(COURTESY_PATTERN, "", regex=True).str.strip()
    df["caption"] = df["caption"].str.replace(FIG_PATTERN, "", regex=True).str.strip()

    # Filter key patterns
    n_key_patterns = 0
    if args.key_patterns:
        key_pattern_mask = df["caption"].str.contains(KEY_PATTERNS, na=False)
        n_key_patterns = key_pattern_mask.sum()
        key_pattern_stems = df[key_pattern_mask]["id"].tolist()
        for stem in key_pattern_stems:
            fpath = os.path.join(dataset_dir, stem + ".jpg")
            os.remove(fpath)
        df = df[~key_pattern_mask]

    df.to_csv(csv_path, index=False)

    print(f"Removed {n_duplicates} duplicate(s)")
    print(f"Removed {n_too_small} too small image(s)")
    print(f"Removed {n_too_big} too big image(s)")
    print(f"Removed {n_no_caption} image(s) with empty caption")
    print(f"Removed {n_multi_subfig} image(s) with multiple sub-figures in caption")
    print(f"Cleaned courtesy attributions from {n_courtesy} caption(s)")
    print(f"Cleaned figure prefixes from {n_fig_prefix} caption(s)")
    print(f"Total rows removed: {n_duplicates + n_too_small + n_too_big + n_no_caption + n_multi_subfig}")
    if args.key_patterns:
        print(f"Removed {n_key_patterns} image(s) matching key patterns")
    print(f"Total rows removed: {n_duplicates + n_too_small + n_too_big + n_no_caption + n_multi_subfig + n_key_patterns}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove duplicate, small, or malformed images from a dataset.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--remove-dir", required=True)
    parser.add_argument("--min-width", type=int, default=90)
    parser.add_argument("--min-height", type=int, default=75)
    parser.add_argument("--key-patterns", action="store_true")
    args = parser.parse_args()
    filter_dataset(args.dataset_dir, args.remove_dir, args.min_width, args.min_height, args.key_patterns)