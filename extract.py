import argparse
import csv
import re
import shutil
from pathlib import Path

import fitz  # PyMuPDF
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox

CAPTION_PATTERN = re.compile(r"^Fig\.?\s{1,5}\d+\.\d+", re.MULTILINE)
CAPTION_SEARCH_MARGIN = 60  # points below image to look for a caption
SUBPART_PATTERN = re.compile(r"\(([A-Z])\)")


def extract_caption_boxes(pdf_path: str, max_pages: int) -> dict[int, list[tuple]]:
    """
    Use PDFMiner to extract all text boxes that start with 'Fig. X.Y',
    grouped by 1-indexed page number.
    Each entry is (x0, y0, x1, y1, text) in PDFMiner coords (bottom-left origin).
    """
    caption_boxes: dict[int, list] = {}
    for page_num, page_layout in enumerate(extract_pages(pdf_path, maxpages=max_pages), start=1):
        boxes = []
        for element in page_layout:
            if not isinstance(element, LTTextBox):
                continue
            text = element.get_text().strip()
            if CAPTION_PATTERN.match(text):
                boxes.append((element.x0, element.y0, element.x1, element.y1, text))
        if boxes:
            caption_boxes[page_num] = boxes
    return caption_boxes


def pdfminer_to_fitz_y(pm_y0: float, pm_y1: float, page_height: float) -> tuple[float, float]:
    """Convert PDFMiner bottom-left y coords to fitz top-left y coords."""
    return page_height - pm_y1, page_height - pm_y0


def find_caption_for_image(
    img_rect: fitz.Rect,
    page_height: float,
    caption_boxes: list[tuple],
) -> str:
    """
    Match an image (fitz coords) to the nearest caption box below it.
    Converts PDFMiner caption boxes to fitz coords for comparison.
    Returns the caption text or empty string if none found within margin.
    """
    best_caption = ""
    best_dist = float("inf")

    for x0, y0, x1, y1, text in caption_boxes:
        fitz_y0, _ = pdfminer_to_fitz_y(y0, y1, page_height)
        dist = fitz_y0 - img_rect.y1
        if 0 <= dist <= CAPTION_SEARCH_MARGIN and dist < best_dist:
            best_caption = " ".join(text.split())
            best_dist = dist

    return best_caption


def split_caption(caption: str) -> tuple[str, list[str]]:
    """
    Split a caption with sub-parts (A), (B), ... into a shared prefix and a list of parts.
    Searches sequentially: finds (A) first, then (B) after that position, etc.
    This avoids false matches like (N) or (I) used as image annotations.
    Returns (prefix, [part1, part2, ...]), or (caption, []) if no sub-parts found.
    """
    matches = []
    search_from = 0
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        pattern = re.compile(r"\(" + letter + r"\)")
        match = pattern.search(caption, search_from)
        if match:
            matches.append(match)
            search_from = match.end()
        else:
            break  # stop as soon as a letter in sequence is not found

    if not matches:
        return caption, []

    prefix = caption[:matches[0].start()].strip()
    parts = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(caption)
        parts.append(caption[start:end].strip())

    return prefix, parts


def assign_subcaptions(images: list[dict], caption: str) -> None:
    """
    For images sharing the same caption, split into sub-parts if present
    and assign each part in top-to-bottom, left-to-right order.
    Falls back to the full caption if no sub-parts or count mismatches.
    """
    prefix, parts = split_caption(caption)
    if not parts or len(parts) != len(images):
        for img in images:
            img["caption"] = caption
        return

    sorted_images = sorted(images, key=lambda im: (im["y"], im["x"]))
    for img, part in zip(sorted_images, parts):
        img["caption"] = f"{prefix} {part}".strip()


RECT_DEDUP_THRESHOLD = 5  # points; rects closer than this are considered duplicates

def is_duplicate_rect(rect: fitz.Rect, seen: list[fitz.Rect]) -> bool:
    """Return True if rect is within RECT_DEDUP_THRESHOLD of any already-seen rect."""
    return any(
        abs(rect.x0 - s.x0) < RECT_DEDUP_THRESHOLD and abs(rect.y0 - s.y0) < RECT_DEDUP_THRESHOLD
        for s in seen
    )


def get_images_on_page(page: fitz.Page, caption_boxes: list[tuple]) -> list[dict]:
    """Return list of images with xref, position (x, y), and caption."""
    page_height = page.rect.height
    images = []

    for img in page.get_images(full=True):
        xref = img[0]
        seen_rects: list[fitz.Rect] = []
        for rect in page.get_image_rects(xref):
            if is_duplicate_rect(rect, seen_rects):
                continue
            seen_rects.append(rect)
            caption = find_caption_for_image(rect, page_height, caption_boxes)
            images.append({
                "xref": xref,
                "x": round(rect.x0, 2),
                "y": round(rect.y0, 2),
                "caption": caption,
            })

    # Group images by shared caption and assign sub-parts
    groups: dict[str, list[dict]] = {}
    for img in images:
        groups.setdefault(img["caption"], []).append(img)

    for caption, group in groups.items():
        if caption:
            assign_subcaptions(group, caption)

    return images


def extract_images_from_pages(
    doc: fitz.Document,
    caption_boxes: dict[int, list[tuple]],
    start: int = 0,
    end: int = 50,
) -> dict[int, list[dict]]:
    """Extract images from pages [start, end) of the document (1-indexed keys)."""
    end = min(end, len(doc))
    result = {}
    for page_num in range(start, end):
        page = doc[page_num]
        boxes = caption_boxes.get(page_num + 1, [])
        images = get_images_on_page(page, boxes)
        if images:
            result[page_num + 1] = images
    return result


def assign_ids(page_images: dict[int, list[dict]]) -> None:
    """Assign global sequential IDs (img_001, img_002, ...) in-place."""
    counter = 1
    for page_num in sorted(page_images):
        for img in page_images[page_num]:
            img["id"] = f"img_{counter:03d}"
            counter += 1


def save_images(doc: fitz.Document, page_images: dict[int, list[dict]], output_dir: Path) -> None:
    """Save each image to output_dir as <id>.jpg."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for page_num, images in page_images.items():
        for img in images:
            pix = fitz.Pixmap(doc, img["xref"])
            if pix.colorspace not in (fitz.csGRAY, fitz.csRGB, fitz.csCMYK):
                pix = fitz.Pixmap(fitz.csRGB, pix)
            if pix.alpha:
                pix = fitz.Pixmap(pix, 0)
            pix.save(output_dir / f"{img['id']}.jpg")


def save_csv(page_images: dict[int, list[dict]], output_dir: Path) -> None:
    """Save a CSV with image id, page, x, y, caption to output_dir/images.csv."""
    csv_path = output_dir / "images.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "page", "x", "y", "caption"])
        writer.writeheader()
        for page_num, images in sorted(page_images.items()):
            for img in images:
                writer.writerow({
                    "id": img["id"],
                    "page": page_num,
                    "x": img["x"],
                    "y": img["y"],
                    "caption": img["caption"],
                })
    print(f"CSV saved to {csv_path}")


def print_results(page_images: dict[int, list[dict]]) -> None:
    """Print image info per page and a total count."""
    total = 0
    for page_num, images in sorted(page_images.items()):
        print(f"Page {page_num}: {len(images)} image(s)")
        for img in images:
            preview = img["caption"][:60] + "..." if len(img["caption"]) > 60 else img["caption"]
            print(f"  {img['id']}  ({img['x']}, {img['y']})  caption: {preview or '—'}")
        total += len(images)
    print(f"\nTotal images found: {total}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract images from a PDF and save metadata.")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("--pages", type=int, default=50, help="Number of pages to scan (default: 50)")
    parser.add_argument("--output-dir", type=Path, default=Path("output_images"), help="Directory to save images and CSV (default: output_images/)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_dir.exists():
        print("Deleting existing output_dir")
        shutil.rmtree(args.output_dir)

    print("Extracting caption boxes with PDFMiner...")
    caption_boxes = extract_caption_boxes(args.pdf, max_pages=args.pages)

    print("Extracting images with fitz...")
    doc = fitz.open(args.pdf)
    page_images = extract_images_from_pages(doc, caption_boxes, start=0, end=args.pages)
    assign_ids(page_images)
    save_images(doc, page_images, args.output_dir)
    save_csv(page_images, args.output_dir)
    doc.close()


if __name__ == "__main__":
    main()