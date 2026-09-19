#!/usr/bin/env python3
"""Cache R-peak-aligned features from the full-view TolerantECG temporal map."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

from protocol import (BASELINE_CAMPAIGN, CAMPAIGN, EMBED_DIM,
                      EXTRACTION_BATCH_SIZE, FEATURE_WINDOW_RADIUS,
                      INTEGRATION_SECONDS, MAX_BEATS, REFINE_SECONDS,
                      REFRACTORY_SECONDS, SAMPLING_RATE, SPLITS, TASKS)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = path.with_suffix(path.suffix + ".incoming")
    incoming.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(incoming, path)


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def load_modern(root: Path):
    source = root / "src/CLEAR-HUG/experiments/q1_modern_mimic_baselines_20260918/prepare_embeddings.py"
    spec = importlib.util.spec_from_file_location("modern_prepare", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_source(root: Path, task: str, split: str, dataset) -> tuple[dict, Path]:
    source = root / "results" / BASELINE_CAMPAIGN / "shared/embeddings/TolerantECG" / task / split
    manifest = json.loads((source / "complete.json").read_text(encoding="utf-8"))
    indices = np.asarray(dataset.indices, dtype=np.int64)
    labels = (np.asarray(dataset.labels[indices]) > 0).astype(np.float32)
    stored_labels = (np.load(source / "labels.npy", mmap_mode="r") > 0).astype(np.float32)
    if manifest["source_indices_sha256"] != array_sha256(indices):
        raise RuntimeError(f"source split indices differ for {task}/{split}")
    if not np.array_equal(labels, stored_labels):
        raise RuntimeError(f"source labels differ for {task}/{split}")
    return manifest, source


def detect_r_peaks(waveform: np.ndarray) -> np.ndarray:
    """Deterministic lightweight Pan-Tompkins-style detector on lead II."""
    signal = np.nan_to_num(np.asarray(waveform[1], dtype=np.float32))
    centered = signal - np.median(signal)
    derivative = np.diff(centered, prepend=centered[0])
    energy = derivative * derivative
    width = max(1, round(INTEGRATION_SECONDS * SAMPLING_RATE))
    integrated = np.convolve(energy, np.ones(width, dtype=np.float32) / width, mode="same")
    maxima = np.flatnonzero((integrated[1:-1] >= integrated[:-2]) &
                            (integrated[1:-1] > integrated[2:])) + 1
    median = float(np.median(integrated))
    mad = float(np.median(np.abs(integrated - median))) + 1e-12
    threshold = max(median + 3.0 * mad, float(np.quantile(integrated, 0.80)))
    candidates = maxima[integrated[maxima] >= threshold]
    if len(candidates) < 2:
        candidates = maxima[integrated[maxima] >= median + 1.5 * mad]
    refractory = max(1, round(REFRACTORY_SECONDS * SAMPLING_RATE))
    selected: list[int] = []
    for candidate in candidates[np.argsort(integrated[candidates])[::-1]]:
        if all(abs(int(candidate) - previous) >= refractory for previous in selected):
            selected.append(int(candidate))
            if len(selected) == MAX_BEATS:
                break
    if not selected:
        selected = [int(np.argmax(integrated))]
    radius = max(1, round(REFINE_SECONDS * SAMPLING_RATE))
    refined = []
    for candidate in selected:
        start, stop = max(0, candidate - radius), min(len(centered), candidate + radius + 1)
        refined.append(start + int(np.argmax(np.abs(centered[start:stop]))))
    return np.asarray(sorted(set(refined))[:MAX_BEATS], dtype=np.int64)


def temporal_map(model: torch.nn.Module, values: torch.Tensor) -> torch.Tensor:
    features = values
    for index in range(4):
        features = model.downsample_layers[index](features)
        features = model.stages[index](features)
    return features


@torch.inference_mode()
def encode_local(model: torch.nn.Module, waveforms: np.ndarray,
                 device: torch.device) -> tuple[np.ndarray, np.ndarray, list[int], int]:
    clean = np.nan_to_num(np.asarray(waveforms, dtype=np.float32),
                          nan=0.0, posinf=0.0, neginf=0.0)
    feature_map = temporal_map(model, torch.from_numpy(clean).to(device))
    map_length, input_length = feature_map.shape[-1], clean.shape[-1]
    local = torch.zeros((len(clean), MAX_BEATS, EMBED_DIM), device=device)
    mask = torch.zeros((len(clean), MAX_BEATS), dtype=torch.bool, device=device)
    counts = []
    for row, waveform in enumerate(clean):
        peaks = detect_r_peaks(waveform)
        counts.append(len(peaks))
        for column, peak in enumerate(peaks):
            center = round(int(peak) * (map_length - 1) / max(1, input_length - 1))
            start = max(0, center - FEATURE_WINDOW_RADIUS)
            stop = min(map_length, center + FEATURE_WINDOW_RADIUS + 1)
            pooled = feature_map[row, :, start:stop].mean(dim=-1)
            local[row, column] = model.norm(pooled)
            mask[row, column] = True
    return local.half().cpu().numpy(), mask.cpu().numpy(), counts, map_length


def extract(root: Path, task: str, device: torch.device,
            batch_size: int = EXTRACTION_BATCH_SIZE, limit: int | None = None) -> None:
    module = load_modern(root)
    model, make_dataset, provenance = module.model_and_provenance(root, "TolerantECG")
    model = model.to(device).eval().requires_grad_(False)
    for split in SPLITS:
        output = root / "results" / CAMPAIGN / "shared/local_features" / task / split
        complete = output / "complete.json"
        if limit is None and complete.is_file():
            value = json.loads(complete.read_text(encoding="utf-8"))
            if value.get("state") == "complete":
                continue
        dataset = make_dataset(task, split)
        source_manifest, _ = verify_source(root, task, split, dataset)
        source_indices = np.asarray(dataset.indices, dtype=np.int64)
        count = len(dataset) if limit is None else min(limit, len(dataset))
        output.mkdir(parents=True, exist_ok=True)
        suffix = "incoming" if limit is None else "canary"
        features_path = output / f"features.npy.{suffix}"
        masks_path = output / f"masks.npy.{suffix}"
        features = open_memmap(features_path, mode="w+", dtype=np.float16,
                               shape=(count, MAX_BEATS, EMBED_DIM))
        masks = open_memmap(masks_path, mode="w+", dtype=np.bool_, shape=(count, MAX_BEATS))
        all_counts: list[int] = []
        map_length = 0
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            waveforms = dataset.ecg[source_indices[start:stop]]
            batch_features, batch_masks, counts, map_length = encode_local(model, waveforms, device)
            features[start:stop], masks[start:stop] = batch_features, batch_masks
            if not np.isfinite(features[start:stop]).all() or not masks[start:stop].any(axis=1).all():
                raise RuntimeError(f"invalid local features for {task}/{split}/{start}:{stop}")
            all_counts.extend(counts)
            features.flush(); masks.flush()
            atomic_json(output / "status.json", {"state": "extracting", "task": task,
                "split": split, "records_done": stop, "records_total": count})
        del features, masks
        if limit is not None:
            continue
        os.replace(features_path, output / "features.npy")
        os.replace(masks_path, output / "masks.npy")
        payload = {"state": "complete", "task": task, "split": split,
            "records": count, "shape": [count, MAX_BEATS, EMBED_DIM], "dtype": "float16",
            "feature_map_length": map_length, "feature_window_radius": FEATURE_WINDOW_RADIUS,
            "sampling_rate": SAMPLING_RATE, "max_beats": MAX_BEATS,
            "beat_count_min": min(all_counts), "beat_count_max": max(all_counts),
            "beat_count_mean": float(np.mean(all_counts)),
            "r_peak_detector": "deterministic lead-II derivative-energy integration",
            "source_indices_sha256": array_sha256(source_indices),
            "source_labels_sha256": source_manifest["labels_sha256"],
            "checkpoint_sha256": provenance["checkpoint_sha256"], "batch_size": batch_size,
            "test_data_loaded": False}
        atomic_json(complete, payload)
        atomic_json(output / "status.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=EXTRACTION_BATCH_SIZE)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    extract(args.root.resolve(), args.task, device, args.batch_size, args.limit)


if __name__ == "__main__":
    main()
