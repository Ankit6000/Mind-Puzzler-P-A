#!/usr/bin/env python3
"""Train browser-readable tile prototypes from labeled crop folders."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps


LABELS = ["00"] + [f"{row}{col}" for row in range(1, 5) for col in range(1, 5)]


def normalise(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    return (values - values.mean()) / (values.std() + 1e-6)


def image_features(path: Path, size: int = 64) -> dict[str, np.ndarray]:
    image = ImageOps.fit(Image.open(path).convert("RGB"), (size, size), method=Image.Resampling.BICUBIC)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray_raw = np.asarray(ImageOps.grayscale(image), dtype=np.float32) / 255.0

    hists = []
    for channel in range(3):
      hist, _ = np.histogram(rgb[:, :, channel], bins=16, range=(0.0, 1.0), density=False)
      hist = hist.astype(np.float32)
      hist /= max(float(hist.sum()), 1.0)
      hists.append(hist)

    edge_raw = np.zeros_like(gray_raw, dtype=np.float32)
    gy = gray_raw[2:, 1:-1] - gray_raw[:-2, 1:-1]
    gx = gray_raw[1:-1, 2:] - gray_raw[1:-1, :-2]
    edge_raw[1:-1, 1:-1] = np.sqrt(gx * gx + gy * gy)

    return {
        "gray": normalise(gray_raw).reshape(-1),
        "edge": normalise(edge_raw).reshape(-1),
        "hist": np.concatenate(hists),
    }


def average_features(features: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: normalise(np.stack([feature[key] for feature in features]).mean(axis=0))
        if key != "hist"
        else np.stack([feature[key] for feature in features]).mean(axis=0)
        for key in ("gray", "edge", "hist")
    }


def vector_distance(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> float:
    total = 0.0
    for key in ("gray", "edge", "hist"):
        diff = left[key] - right[key]
        total += float(np.dot(diff, diff))
    return total


def kmeans(features: list[dict[str, np.ndarray]], k: int, iterations: int = 18) -> list[tuple[int, dict[str, np.ndarray]]]:
    if len(features) <= k:
        return [(1, feature) for feature in features]

    centres = [features[0]]
    while len(centres) < k:
        candidate = max(features, key=lambda feature: min(vector_distance(feature, centre) for centre in centres))
        centres.append(candidate)

    assignments = [0] * len(features)
    for _ in range(iterations):
        changed = False
        for index, feature in enumerate(features):
            cluster = min(range(len(centres)), key=lambda centre_index: vector_distance(feature, centres[centre_index]))
            if cluster != assignments[index]:
                assignments[index] = cluster
                changed = True
        if not changed:
            break

        next_centres = []
        for cluster in range(k):
            cluster_features = [feature for index, feature in enumerate(features) if assignments[index] == cluster]
            if cluster_features:
                next_centres.append(average_features(cluster_features))
            else:
                next_centres.append(centres[cluster])
        centres = next_centres

    output = []
    for cluster, centre in enumerate(centres):
        count = sum(1 for assignment in assignments if assignment == cluster)
        if count:
            output.append((count, centre))
    return output


def rounded(values: Iterable[float]) -> list[float]:
    return [round(float(value), 5) for value in values]


def train(samples_dir: Path, output_path: Path, max_prototypes: int, size: int) -> dict[str, object]:
    prototypes: dict[str, list[dict[str, object]]] = {}
    sample_count = 0
    label_counts: dict[str, int] = {}

    for label in LABELS:
        label_dir = samples_dir / label
        if not label_dir.exists():
            continue
        files = sorted(
            path
            for path in label_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if not files:
            continue
        features = [image_features(path, size=size) for path in files]
        label_counts[label] = len(features)
        sample_count += len(features)
        clusters = kmeans(features, k=min(max_prototypes, len(features)))
        prototypes[label] = [
            {
                "count": count,
                "gray": rounded(feature["gray"]),
                "edge": rounded(feature["edge"]),
                "hist": rounded(feature["hist"]),
            }
            for count, feature in clusters
        ]

    model = {
        "version": 1,
        "kind": "tile-prototype-model",
        "feature": "gray-edge-hist-v1",
        "size": size,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "sampleCount": sample_count,
        "labelCounts": label_counts,
        "maxPrototypes": max_prototypes,
        "prototypes": prototypes,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train static detector prototypes for GitHub Pages.")
    parser.add_argument("--samples", default="training/samples", type=Path, help="Folder with one subfolder per label.")
    parser.add_argument("--out", default="static/trained-model.json", type=Path, help="Output JSON model used by the PWA.")
    parser.add_argument("--max-prototypes", default=4, type=int, help="Prototype clusters per class.")
    parser.add_argument("--size", default=64, type=int, help="Feature image size. Must match puzzle-core.js.")
    args = parser.parse_args()

    model = train(args.samples, args.out, args.max_prototypes, args.size)
    print(f"Wrote {args.out} with {model['sampleCount']} sample(s) across {len(model['prototypes'])} class(es).")


if __name__ == "__main__":
    main()
