#!/usr/bin/env python3
"""Cache contextual lead deltas g(x)-g(x without lead l), validation only."""
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
                      EXTRACTION_BATCH_SIZE, LEADS, SPLITS, TASKS)


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


@torch.inference_mode()
def encode_leave_one_out(model: torch.nn.Module, waveforms: np.ndarray,
                         device: torch.device) -> np.ndarray:
    clean = np.nan_to_num(np.asarray(waveforms, dtype=np.float32),
                          nan=0.0, posinf=0.0, neginf=0.0)
    values = torch.from_numpy(clean).to(device)
    batch = len(values)
    views = values[:, None, :, :].expand(-1, LEADS, -1, -1).clone()
    lead_ids = torch.arange(LEADS, device=device)
    views[:, lead_ids, lead_ids, :] = 0
    features = model(views.reshape(batch * LEADS, LEADS, values.shape[-1]))
    return features.reshape(batch, LEADS, EMBED_DIM).cpu().numpy()


def extract(root: Path, task: str, device: torch.device,
            batch_size: int = EXTRACTION_BATCH_SIZE, limit: int | None = None) -> None:
    module = load_modern(root)
    model, make_dataset, provenance = module.model_and_provenance(root, "TolerantECG")
    model = model.to(device).eval().requires_grad_(False)
    for split in SPLITS:
        output = root / "results" / CAMPAIGN / "shared/delta_features" / task / split
        complete = output / "complete.json"
        if limit is None and complete.is_file():
            value = json.loads(complete.read_text(encoding="utf-8"))
            if value.get("state") == "complete":
                continue
        dataset = make_dataset(task, split)
        source_manifest, source = verify_source(root, task, split, dataset)
        global_features = np.load(source / "features.npy", mmap_mode="r")
        source_indices = np.asarray(dataset.indices, dtype=np.int64)
        count = len(dataset) if limit is None else min(limit, len(dataset))
        output.mkdir(parents=True, exist_ok=True)
        path = output / ("features.npy.incoming" if limit is None else "canary.npy")
        features = open_memmap(path, mode="w+", dtype=np.float16,
                               shape=(count, LEADS, EMBED_DIM))
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            waveforms = dataset.ecg[source_indices[start:stop]]
            leave_out = encode_leave_one_out(model, waveforms, device)
            delta = np.asarray(global_features[start:stop], dtype=np.float32)[:, None, :] - leave_out
            features[start:stop] = delta.astype(np.float16)
            if not np.isfinite(features[start:stop]).all():
                raise RuntimeError(f"non-finite delta features for {task}/{split}/{start}:{stop}")
            features.flush()
            atomic_json(output / "status.json", {"state": "extracting", "task": task,
                "split": split, "records_done": stop, "records_total": count})
        del features
        if limit is not None:
            continue
        os.replace(path, output / "features.npy")
        payload = {"state": "complete", "task": task, "split": split,
            "records": count, "shape": [count, LEADS, EMBED_DIM], "dtype": "float16",
            "view": "global full-view embedding minus leave-one-lead-out embedding",
            "source_indices_sha256": array_sha256(source_indices),
            "source_labels_sha256": source_manifest["labels_sha256"],
            "checkpoint_sha256": provenance["checkpoint_sha256"], "batch_size": batch_size,
            "nonfinite_waveform_policy": "replace NaN and +/-Inf with zero before encoder"}
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
