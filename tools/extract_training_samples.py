#!/usr/bin/env python3
"""Extract labeled tile crops from puzzle photos.

labels.csv format:
filename,cell1,cell2,...,cell16
photo1.jpg,11,12,13,14,21,22,23,24,31,32,33,34,41,42,43,44
scramble.jpg,23,11,22,24,12,00,31,13,32,33,14,41,42,21,44,43
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app


VALID_LABELS = {"00", *app.LABELS}


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(value).stem).strip("_") or "photo"


def read_rows(labels_csv: Path) -> list[tuple[str, list[str]]]:
    rows: list[tuple[str, list[str]]] = []
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for line_number, row in enumerate(reader, start=1):
            if not row or not row[0].strip() or row[0].strip().startswith("#"):
                continue
            if row[0].strip().lower() == "filename":
                continue
            filename = row[0].strip()
            labels = [cell.strip() for cell in row[1:]]
            if len(labels) != 16:
                raise ValueError(f"{labels_csv}:{line_number}: expected 16 labels after filename.")
            invalid = [label for label in labels if label not in VALID_LABELS]
            if invalid:
                raise ValueError(f"{labels_csv}:{line_number}: invalid label(s): {', '.join(invalid)}")
            rows.append((filename, labels))
    return rows


def extract_samples(photo_dir: Path, labels_csv: Path, output_dir: Path) -> int:
    rows = read_rows(labels_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for filename, labels in rows:
        photo_path = photo_dir / filename
        if not photo_path.exists():
            raise FileNotFoundError(f"Photo not found: {photo_path}")

        image = Image.open(photo_path).convert("RGB")
        board, _crop_box = app.crop_board(image)
        photo_stem = safe_stem(filename)

        for index, label in enumerate(labels):
            label_dir = output_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            crop = app.crop_cell(board, index, margin_ratio=0.055)
            crop.save(label_dir / f"{photo_stem}_{index + 1:02d}.png")
            written += 1

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Create labeled training crops from puzzle photos.")
    parser.add_argument("--photos", default="training/photos", type=Path, help="Folder containing labeled puzzle photos.")
    parser.add_argument("--labels", default="training/labels.csv", type=Path, help="CSV mapping photos to 16 labels.")
    parser.add_argument("--out", default="training/samples", type=Path, help="Output folder for class-labeled crops.")
    args = parser.parse_args()

    count = extract_samples(args.photos, args.labels, args.out)
    print(f"Extracted {count} labeled tile crop(s) into {args.out}")


if __name__ == "__main__":
    main()
