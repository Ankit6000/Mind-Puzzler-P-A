#!/usr/bin/env python3
"""Camera app for detecting and solving the outside-slot puzzle."""

from __future__ import annotations

import base64
import io
import os
import threading
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, ImageOps

import puzzle_solver


ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE_DIR = ROOT / "reference"
if not DEFAULT_REFERENCE_DIR.exists():
    DEFAULT_REFERENCE_DIR = Path(r"F:\puzzle correct")
REFERENCE_DIR = Path(os.environ.get("PUZZLE_REFERENCE_DIR", str(DEFAULT_REFERENCE_DIR)))
LABELS = [f"{row}{col}" for row in range(1, 5) for col in range(1, 5)]
TILES = [puzzle_solver.LABEL_TO_TILE[label] for label in LABELS]
TILE_LABEL_BY_VALUE = {puzzle_solver.LABEL_TO_TILE[label]: label for label in LABELS}
ALGORITHMS = [
    {"value": "fast", "label": "Fast solver (default)"},
    {"value": "auto", "label": "Auto + exact fallback"},
    {"value": "ida-star", "label": "IDA* (exact, slower)"},
    {"value": "a-star-closed", "label": "A* (exact, memory heavy)"},
    {"value": "bfs", "label": "BFS (tiny only)"},
]

app = Flask(__name__)

SOLVE_JOBS: dict[str, dict[str, Any]] = {}
SOLVE_JOB_LOCK = threading.Lock()
SOLVE_JOB_TTL_SECONDS = 60 * 30


@dataclass(frozen=True)
class ImageFeatures:
    gray: np.ndarray
    edge: np.ndarray
    hist: np.ndarray


@dataclass(frozen=True)
class CellMetrics:
    colour_coverage: float
    white_coverage: float
    dark_coverage: float
    mean_brightness: float
    mean_saturation: float
    texture: float
    blank_score: float


def decode_data_url(data_url: str) -> Image.Image:
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError("Image data is not valid base64.") from exc

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # Pillow raises several image-specific exceptions.
        raise ValueError("Image could not be opened.") from exc

    if image.width < 180 or image.height < 180:
        raise ValueError("Captured image is too small. Move the camera closer and retake it.")
    return image


def center_crop_square(image: Image.Image) -> Image.Image:
    side = min(image.width, image.height)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return image.crop((left, top, left + side, top + side))


def smooth_projection(values: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="same")


def longest_segment(active: np.ndarray, minimum: int) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    start: int | None = None

    for index, value in enumerate(active):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= minimum and (best is None or index - start > best[1] - best[0]):
                best = (start, index)
            start = None

    if start is not None:
        index = len(active)
        if index - start >= minimum and (best is None or index - start > best[1] - best[0]):
            best = (start, index)

    return best


def expand_to_square(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    crop_w = right - left
    crop_h = bottom - top
    side = max(crop_w, crop_h)
    cx = (left + right) / 2
    cy = (top + bottom) / 2

    left = int(round(cx - side / 2))
    top = int(round(cy - side / 2))
    right = left + side
    bottom = top + side

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left -= right - width
        right = width
    if bottom > height:
        top -= bottom - height
        bottom = height

    return max(0, left), max(0, top), min(width, right), min(height, bottom)


def detect_board_box(image: Image.Image) -> tuple[int, int, int, int] | None:
    width, height = image.size
    longest_side = max(width, height)
    scale = 900 / longest_side if longest_side > 900 else 1.0
    work = image
    if scale < 1.0:
        work = image.resize((int(width * scale), int(height * scale)), Image.Resampling.BILINEAR)

    arr = np.asarray(work.convert("RGB"), dtype=np.float32) / 255.0
    rgb_max = arr.max(axis=2)
    rgb_min = arr.min(axis=2)
    saturation = (rgb_max - rgb_min) / (rgb_max + 1e-6)

    # The puzzle tiles are strongly coloured; the plastic frame and table are not.
    mask = (saturation > 0.22) & (rgb_max > 0.20)
    if mask.mean() < 0.02:
        return None

    row_projection = smooth_projection(mask.mean(axis=1), max(5, work.height // 80))
    row_threshold = max(0.12, float(row_projection.max()) * 0.36)
    y_segment = longest_segment(row_projection > row_threshold, int(work.height * 0.28))
    if y_segment is None:
        return None

    y1, y2 = y_segment
    board_rows = mask[y1:y2, :]
    col_projection = smooth_projection(board_rows.mean(axis=0), max(5, work.width // 80))
    col_threshold = max(0.10, float(col_projection.max()) * 0.34)
    x_segment = longest_segment(col_projection > col_threshold, int(work.width * 0.28))
    if x_segment is None:
        return None

    x1, x2 = x_segment
    pad_x = int((x2 - x1) * 0.012)
    pad_y = int((y2 - y1) * 0.012)
    x1 = max(0, x1 - pad_x)
    x2 = min(work.width, x2 + pad_x)
    y1 = max(0, y1 - pad_y)
    y2 = min(work.height, y2 + pad_y)

    if scale != 1.0:
        x1 = int(round(x1 / scale))
        x2 = int(round(x2 / scale))
        y1 = int(round(y1 / scale))
        y2 = int(round(y2 / scale))

    box = expand_to_square((x1, y1, x2, y2), width, height)
    side = min(box[2] - box[0], box[3] - box[1])
    if side < min(width, height) * 0.25:
        return None
    return box


def crop_board(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    box = detect_board_box(image)
    if box is None:
        return center_crop_square(image), None
    return image.crop(box), box


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def crop_cell(board: Image.Image, index: int, margin_ratio: float = 0.055) -> Image.Image:
    row, col = divmod(index, 4)
    cell_w = board.width / 4
    cell_h = board.height / 4
    margin_x = cell_w * margin_ratio
    margin_y = cell_h * margin_ratio
    left = int(col * cell_w + margin_x)
    top = int(row * cell_h + margin_y)
    right = int((col + 1) * cell_w - margin_x)
    bottom = int((row + 1) * cell_h - margin_y)
    return board.crop((left, top, right, bottom))


def normalise_array(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    return (values - values.mean()) / (values.std() + 1e-6)


def edge_array(gray: np.ndarray) -> np.ndarray:
    gy = np.zeros_like(gray)
    gx = np.zeros_like(gray)
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    return normalise_array(np.sqrt(gx * gx + gy * gy))


def image_features(image: Image.Image, size: int = 96) -> ImageFeatures:
    resized = ImageOps.fit(image.convert("RGB"), (size, size), method=Image.Resampling.BICUBIC)
    rgb = np.asarray(resized, dtype=np.float32) / 255.0
    gray = np.asarray(ImageOps.grayscale(resized), dtype=np.float32) / 255.0
    gray = normalise_array(gray)

    hists = []
    for channel in range(3):
        hist, _ = np.histogram(rgb[:, :, channel], bins=16, range=(0.0, 1.0), density=False)
        hist = hist.astype(np.float32)
        hist /= max(float(hist.sum()), 1.0)
        hists.append(hist)

    return ImageFeatures(gray=gray.ravel(), edge=edge_array(gray).ravel(), hist=np.concatenate(hists))


def cell_metrics(image: Image.Image, size: int = 96) -> CellMetrics:
    resized = ImageOps.fit(image.convert("RGB"), (size, size), method=Image.Resampling.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    rgb_max = arr.max(axis=2)
    rgb_min = arr.min(axis=2)
    saturation = (rgb_max - rgb_min) / (rgb_max + 1e-6)
    gray = np.asarray(ImageOps.grayscale(resized), dtype=np.float32) / 255.0
    gy = np.zeros_like(gray)
    gx = np.zeros_like(gray)
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    texture = float(np.sqrt(gx * gx + gy * gy).mean())

    colour_coverage = float(((saturation > 0.20) & (rgb_max > 0.24)).mean())
    white_coverage = float(((saturation < 0.16) & (rgb_max > 0.58)).mean())
    dark_coverage = float((rgb_max < 0.30).mean())
    mean_brightness = float(rgb_max.mean())
    mean_saturation = float(saturation.mean())

    blank_score = (
        0.46 * white_coverage
        + 0.34 * (1.0 - colour_coverage)
        + 0.13 * (1.0 - mean_saturation)
        + 0.07 * mean_brightness
        - 0.18 * dark_coverage
        - 0.20 * texture
    )
    blank_score = max(0.0, min(1.0, blank_score))

    return CellMetrics(
        colour_coverage=colour_coverage,
        white_coverage=white_coverage,
        dark_coverage=dark_coverage,
        mean_brightness=mean_brightness,
        mean_saturation=mean_saturation,
        texture=texture,
        blank_score=blank_score,
    )


def correlation_cost(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right)) + 1e-6
    corr = float(np.dot(left, right) / denom)
    return max(0.0, min(1.0, (1.0 - corr) / 2.0))


def feature_cost(left: ImageFeatures, right: ImageFeatures) -> float:
    gray = correlation_cost(left.gray, right.gray)
    edge = correlation_cost(left.edge, right.edge)
    hist = float(np.abs(left.hist - right.hist).sum() / 6.0)
    return 0.58 * gray + 0.27 * edge + 0.15 * hist


@lru_cache(maxsize=1)
def reference_features() -> dict[str, ImageFeatures]:
    refs: dict[str, ImageFeatures] = {}
    missing: list[str] = []
    for label in LABELS:
        path = REFERENCE_DIR / f"{label}.png"
        if not path.exists():
            missing.append(str(path))
            continue
        refs[label] = image_features(Image.open(path).convert("RGB"))

    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"Missing reference image(s):\n{joined}")
    return refs


def best_candidates(features: ImageFeatures) -> list[dict[str, Any]]:
    candidates = []
    for label, ref_features in reference_features().items():
        candidates.append({"label": label, "cost": feature_cost(features, ref_features)})
    candidates.sort(key=lambda item: item["cost"])
    return candidates


def best_candidates_from_variants(features: list[ImageFeatures]) -> list[dict[str, Any]]:
    candidates = []
    for label, ref_features in reference_features().items():
        best_cost = min(feature_cost(feature, ref_features) for feature in features)
        candidates.append({"label": label, "cost": best_cost})
    candidates.sort(key=lambda item: item["cost"])
    return candidates


def solve_assignment(cost_rows: list[dict[int, float]]) -> tuple[float, list[int]]:
    states: dict[int, tuple[float, list[int]]] = {0: (0.0, [])}

    for row_costs in cost_rows:
        next_states: dict[int, tuple[float, list[int]]] = {}
        for mask, (total, labels) in states.items():
            for tile in TILES:
                bit = 1 << (tile - 1)
                if mask & bit:
                    continue
                new_mask = mask | bit
                new_total = total + row_costs[tile]
                current = next_states.get(new_mask)
                if current is None or new_total < current[0]:
                    next_states[new_mask] = (new_total, labels + [tile])
        states = next_states

    best_mask, best_state = min(states.items(), key=lambda item: item[1][0])
    return best_state[0], best_state[1]


def detect_blank_index(
    metrics: list[CellMetrics],
    ranked: list[list[dict[str, Any]]],
    assigned_costs: list[float],
) -> int | None:
    if not metrics:
        return None

    median_cost = float(np.median(assigned_costs))
    ordered = sorted(range(len(metrics)), key=lambda index: metrics[index].blank_score, reverse=True)
    best_index = ordered[0]
    best_metrics = metrics[best_index]
    second_score = metrics[ordered[1]].blank_score if len(ordered) > 1 else 0.0
    best_gap = ranked[best_index][1]["cost"] - ranked[best_index][0]["cost"] if len(ranked[best_index]) > 1 else 0.0
    best_tile_cost = assigned_costs[best_index]

    # Strong blank: a plain white/cream cell with almost no puzzle artwork.
    if (
        best_metrics.blank_score >= 0.66
        and best_metrics.colour_coverage <= 0.30
        and best_metrics.white_coverage >= 0.48
        and best_metrics.blank_score - second_score >= 0.08
    ):
        return best_index

    # Softer blank: still visually blank, and the numbered-tile matcher dislikes it.
    if (
        best_metrics.blank_score >= 0.56
        and best_metrics.colour_coverage <= 0.42
        and best_tile_cost > max(0.14, median_cost * 1.25)
    ):
        return best_index

    # Fallback for non-white empty holes: very poor and ambiguous tile match.
    worst_index = int(np.argmax(assigned_costs))
    worst_cost = assigned_costs[worst_index]
    worst_two = ranked[worst_index][:2]
    worst_gap = float(worst_two[1]["cost"] - worst_two[0]["cost"]) if len(worst_two) > 1 else 0.0
    if (
        worst_cost > max(0.18, median_cost * 1.55)
        and worst_gap < 0.055
        and metrics[worst_index].blank_score > 0.38
    ):
        return worst_index

    # The gap value is intentionally unused in the final decision, but keeping
    # the local name around makes tuning easier while debugging real photos.
    _ = best_gap
    return None


def analyse_board(board: Image.Image) -> dict[str, Any]:
    board, crop_box = crop_board(board)
    cells = [crop_cell(board, index) for index in range(16)]
    metric_cells = [crop_cell(board, index, margin_ratio=0.11) for index in range(16)]
    metrics = [cell_metrics(cell) for cell in metric_cells]
    cell_feature_variants = [
        [image_features(crop_cell(board, index, margin_ratio=margin)) for margin in (0.035, 0.055, 0.08)]
        for index in range(16)
    ]
    ranked = [best_candidates_from_variants(features) for features in cell_feature_variants]

    cost_rows = []
    for candidates in ranked:
        by_tile = {
            puzzle_solver.LABEL_TO_TILE[candidate["label"]]: float(candidate["cost"])
            for candidate in candidates
        }
        cost_rows.append(by_tile)

    _, all_tile_assignment = solve_assignment(cost_rows)
    assigned_costs = [cost_rows[index][tile] for index, tile in enumerate(all_tile_assignment)]
    blank_index = detect_blank_index(metrics, ranked, assigned_costs)

    labels: list[str]
    if blank_index is not None:
        reduced_rows = [row for index, row in enumerate(cost_rows) if index != blank_index]
        _, reduced_assignment = solve_assignment(reduced_rows)
        labels = []
        assignment_iter = iter(reduced_assignment)
        for index in range(16):
            if index == blank_index:
                labels.append("00")
            else:
                labels.append(TILE_LABEL_BY_VALUE[next(assignment_iter)])
    else:
        labels = [TILE_LABEL_BY_VALUE[tile] for tile in all_tile_assignment]

    detected = []
    for index, label in enumerate(labels):
        candidates = ranked[index][:4]
        best = candidates[0]["cost"]
        second = candidates[1]["cost"] if len(candidates) > 1 else best
        confidence = max(0.0, min(1.0, (second - best) / 0.16))
        if label == "00":
            confidence = metrics[index].blank_score
        detected.append(
            {
                "index": index,
                "row": index // 4 + 1,
                "col": index % 4 + 1,
                "label": label,
                "confidence": round(confidence, 3),
                "blankScore": round(metrics[index].blank_score, 3),
                "colourCoverage": round(metrics[index].colour_coverage, 3),
                "candidates": [
                    {"label": candidate["label"], "score": round(1.0 - float(candidate["cost"]), 3)}
                    for candidate in candidates
                ],
            }
        )

    missing_tiles = sorted(set(LABELS) - {label for label in labels if label != "00"})
    outside = "00" if not missing_tiles else missing_tiles[0]

    return {
        "cells": detected,
        "labels": labels,
        "outside": outside,
        "blankIndex": blank_index,
        "referenceDir": str(REFERENCE_DIR),
        "boardImage": image_to_data_url(board),
        "cropBox": crop_box,
    }


def state_from_grid_labels(labels: list[str]) -> tuple[tuple[int, ...], str]:
    if len(labels) != 16:
        raise ValueError("Expected 16 grid labels.")

    normalised = [label.strip() for label in labels]
    invalid = [label for label in normalised if label not in LABELS and label != "00"]
    if invalid:
        raise ValueError(f"Unknown label(s): {', '.join(invalid)}")

    blank_count = normalised.count("00")
    if blank_count > 1:
        raise ValueError("Only one grid position can be 00.")

    tile_labels = [label for label in normalised if label != "00"]
    duplicates = sorted({label for label in tile_labels if tile_labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"Duplicate tile(s): {', '.join(duplicates)}")

    missing = sorted(set(LABELS) - set(tile_labels))
    if blank_count == 0:
        if missing:
            raise ValueError(f"Missing tile(s): {', '.join(missing)}")
        outside = "00"
    else:
        if len(missing) != 1:
            raise ValueError("When 00 is in the grid, exactly one numbered tile must be outside.")
        outside = missing[0]

    tokens = normalised + [outside]
    return puzzle_solver.parse_state(tokens), outside


def state_to_labels(state: tuple[int, ...]) -> list[str]:
    return [puzzle_solver.TILE_LABELS[tile] for tile in state]


def solve_state(state: tuple[int, ...], outside: str, algorithm: str) -> dict[str, Any]:
    algorithm_id = puzzle_solver.normalise_algorithm(algorithm)
    max_nodes = (
        250_000
        if algorithm_id == "bfs"
        else 1_000_000
        if algorithm_id == "a-star-closed"
        else 1_500_000
        if algorithm_id == "fast"
        else None
    )
    max_seconds = 30 if algorithm_id == "fast" else None
    moves, stats = puzzle_solver.solve(state, max_seconds=max_seconds, algorithm=algorithm, max_nodes=max_nodes)
    states = [state_to_labels(step_state) for step_state in puzzle_solver.apply_moves(state, moves)]
    move_details = puzzle_solver.describe_moves(state, moves)
    return {
        "outside": outside,
        "lowerBound": puzzle_solver.heuristic(state),
        "moves": [puzzle_solver.TILE_LABELS[tile] for tile in moves],
        "moveDetails": move_details,
        "states": states,
        "stats": {
            "nodes": stats.nodes,
            "iterations": stats.iterations,
            "seconds": round(stats.seconds, 3),
            "algorithm": stats.algorithm,
        },
    }


def solve_labels(labels: list[str], algorithm: str) -> dict[str, Any]:
    state, outside = state_from_grid_labels(labels)
    return solve_state(state, outside, algorithm)


def cleanup_solve_jobs() -> None:
    cutoff = time.time() - SOLVE_JOB_TTL_SECONDS
    with SOLVE_JOB_LOCK:
        stale_ids = [
            job_id
            for job_id, job in SOLVE_JOBS.items()
            if float(job.get("updatedAt", job.get("startedAt", 0))) < cutoff
        ]
        for job_id in stale_ids:
            SOLVE_JOBS.pop(job_id, None)


def run_solve_job(job_id: str, labels: list[str], algorithm: str) -> None:
    with SOLVE_JOB_LOCK:
        if job_id in SOLVE_JOBS:
            SOLVE_JOBS[job_id].update(
                {
                    "status": "running",
                    "progress": 15,
                    "message": "Checking board",
                    "updatedAt": time.time(),
                }
            )

    try:
        state, outside = state_from_grid_labels(labels)
        lower_bound = puzzle_solver.heuristic(state)
        with SOLVE_JOB_LOCK:
            if job_id in SOLVE_JOBS:
                SOLVE_JOBS[job_id].update(
                    {
                        "progress": 35,
                        "message": f"Searching ({lower_bound}+ moves minimum)",
                        "lowerBound": lower_bound,
                        "outside": outside,
                        "updatedAt": time.time(),
                    }
                )

        result = solve_state(state, outside, algorithm)

        with SOLVE_JOB_LOCK:
            if job_id in SOLVE_JOBS:
                started_at = float(SOLVE_JOBS[job_id].get("startedAt", time.time()))
                SOLVE_JOBS[job_id].update(
                    {
                        "status": "done",
                        "progress": 100,
                        "message": "Solution ready",
                        "elapsed": round(time.time() - started_at, 2),
                        "result": result,
                        "updatedAt": time.time(),
                    }
                )
    except Exception as exc:
        with SOLVE_JOB_LOCK:
            if job_id in SOLVE_JOBS:
                started_at = float(SOLVE_JOBS[job_id].get("startedAt", time.time()))
                SOLVE_JOBS[job_id].update(
                    {
                        "status": "error",
                        "progress": 100,
                        "message": str(exc),
                        "elapsed": round(time.time() - started_at, 2),
                        "updatedAt": time.time(),
                    }
                )


@app.get("/")
def index() -> str:
    return render_template("index.html", labels=["00"] + LABELS, algorithms=ALGORITHMS)


@app.get("/reference/<path:filename>")
def reference_asset(filename: str) -> Any:
    return send_from_directory(REFERENCE_DIR, filename)


@app.get("/api/health")
def health() -> Any:
    return jsonify({"ok": True, "referenceDir": str(REFERENCE_DIR), "labels": LABELS, "algorithms": ALGORITHMS})


@app.post("/api/analyze")
def analyze() -> Any:
    payload = request.get_json(silent=True) or {}
    data_url = payload.get("image")
    if not data_url:
        return jsonify({"error": "No image supplied."}), 400

    try:
        board = decode_data_url(str(data_url))
        return jsonify(analyse_board(board))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/solve")
def solve() -> Any:
    payload = request.get_json(silent=True) or {}
    labels = payload.get("labels")
    if not isinstance(labels, list):
        return jsonify({"error": "Expected labels array."}), 400

    try:
        algorithm = str(payload.get("algorithm") or "fast")
        return jsonify(solve_labels([str(label) for label in labels], algorithm))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/solve/start")
def start_solve() -> Any:
    cleanup_solve_jobs()
    payload = request.get_json(silent=True) or {}
    labels = payload.get("labels")
    if not isinstance(labels, list):
        return jsonify({"error": "Expected labels array."}), 400

    algorithm = str(payload.get("algorithm") or "fast")
    label_values = [str(label) for label in labels]
    job_id = uuid.uuid4().hex
    now = time.time()
    with SOLVE_JOB_LOCK:
        SOLVE_JOBS[job_id] = {
            "id": job_id,
            "status": "running",
            "progress": 5,
            "message": "Queued",
            "startedAt": now,
            "updatedAt": now,
            "elapsed": 0,
        }

    thread = threading.Thread(target=run_solve_job, args=(job_id, label_values, algorithm), daemon=True)
    thread.start()
    return jsonify({"jobId": job_id, "status": "running", "progress": 5, "message": "Queued"})


@app.get("/api/solve/status/<job_id>")
def solve_status(job_id: str) -> Any:
    with SOLVE_JOB_LOCK:
        job = SOLVE_JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Solve job not found."}), 404
        payload = dict(job)

    if payload.get("status") == "running":
        elapsed = time.time() - float(payload.get("startedAt", time.time()))
        payload["elapsed"] = round(elapsed, 2)
        if elapsed > 20 and "Searching" in str(payload.get("message", "")):
            payload["message"] = "Still searching - check detected labels if this was a light scramble"
    return jsonify(payload)


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PUZZLE_APP_PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG") == "1",
        use_reloader=False,
    )
