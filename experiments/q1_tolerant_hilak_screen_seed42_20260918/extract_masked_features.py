#!/usr/bin/env python3
"""Cache keep-one-lead TolerantECG features in exact MERL train/val order."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap

from protocol import (BASELINE_CAMPAIGN, CAMPAIGN, EMBED_DIM,
                      EXTRACTION_BATCH_SIZE, LEADS, SPLITS, TASKS)


TASK_CONFIG = {
    "superdiagnostic": ("PTBXL_super", "ptbxl", "ptbxl_diagnostic_class_metadata_final.csv", "ptbxl/super_class", "ptbxl_super_class", 6),
    "form": ("PTBXL_form", "ptbxl", "ptbxl_form_metadata_final.csv", "ptbxl/form", "ptbxl_form", 6),
    "cpsc2018": ("CPSC", "cpsc", "cpsc2018_metadata_final.csv", "icbeb", "icbeb", 7),
    "csn": ("CSN", "csn", "csn_metadata_final.csv", "chapman", "chapman", 3),
}


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


def record_ids(kind: str, frame: pd.DataFrame) -> np.ndarray:
    if kind == "ptbxl":
        return frame["ecg_id"].map(lambda value: str(int(value))).to_numpy(dtype=str)
    if kind == "csn":
        return frame["ecg_path"].map(lambda value: Path(str(value)).stem).to_numpy(dtype=str)
    return frame["filename"].astype(str).to_numpy()


def merl_indices(root: Path, task: str, split: str, dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset_name, kind, metadata_name, split_dir, prefix, meta_columns = TASK_CONFIG[task]
    split_root = root / "src/CLEAR-HUG/experiments/q1_ked_tolerant_merl_protocol_20260918/data_split"
    frame = pd.read_csv(split_root / split_dir / f"{prefix}_{split}.csv")
    ids = record_ids(kind, frame)
    labels = frame.iloc[:, meta_columns:].to_numpy(dtype=np.float32)
    if kind == "cpsc":
        indices = frame["ecg_id"].to_numpy(dtype=np.int64) - 1
    elif kind == "csn":
        raw_root = root / "src/CLEAR-HUG/datasets/dataset_preprocess/CSN/WFDBRecords"
        names = [path.stem for path in sorted(raw_root.glob("**/*.hea"))]
        if len(names) != len(dataset.ecg) or len(set(names)) != len(names):
            raise RuntimeError("CSN raw inventory does not match the processed waveform array")
        lookup = {name: index for index, name in enumerate(names)}
        indices = np.asarray([lookup[value] for value in ids], dtype=np.int64)
    else:
        processed = root / "results" / "q1-ecgfix-clocs-six-task-three-fraction-seeds42-46-55-20260916" / "shared/processed"
        metadata = pd.read_csv(processed / metadata_name)
        lookup = dict(zip(metadata["ecg_id"].map(lambda value: str(int(value))),
                          metadata["ecg_index"].astype(np.int64)))
        indices = np.asarray([lookup[value] for value in ids], dtype=np.int64)
    if int(indices.max()) >= len(dataset.ecg):
        raise RuntimeError(f"waveform index out of bounds for {task}/{split}")
    return indices, ids, labels


def verify_against_baseline(root: Path, task: str, split: str,
                            ids: np.ndarray, labels: np.ndarray) -> dict:
    folder = root / "results" / BASELINE_CAMPAIGN / "shared/merl_embeddings/TolerantECG" / task / split
    expected_ids = np.load(folder / "record_ids.npy", mmap_mode="r")
    expected_labels = np.load(folder / "labels.npy", mmap_mode="r")
    if not np.array_equal(ids, expected_ids):
        raise RuntimeError(f"record order differs from baseline for {task}/{split}")
    if not np.array_equal(labels, expected_labels):
        raise RuntimeError(f"labels differ from baseline for {task}/{split}")
    return json.loads((folder / "complete.json").read_text(encoding="utf-8"))


@torch.inference_mode()
def encode_masked(model: torch.nn.Module, waveforms: np.ndarray,
                  device: torch.device) -> np.ndarray:
    # Some processed CSN records retain non-finite source samples. The audited
    # baseline extraction sanitizes recovered raw records before inference; do
    # the same for every masked view so a single bad sample cannot poison an
    # entire ConvNeXt feature vector.
    clean = np.nan_to_num(np.asarray(waveforms, dtype=np.float32),
                          nan=0.0, posinf=0.0, neginf=0.0)
    values = torch.from_numpy(clean).to(device)
    batch = len(values)
    masked = torch.zeros((batch, LEADS, LEADS, values.shape[-1]), device=device)
    lead_ids = torch.arange(LEADS, device=device)
    masked[:, lead_ids, lead_ids, :] = values[:, lead_ids, :]
    features = model(masked.reshape(batch * LEADS, LEADS, values.shape[-1]))
    return features.reshape(batch, LEADS, EMBED_DIM).half().cpu().numpy()


def extract(root: Path, task: str, device: torch.device,
            batch_size: int = EXTRACTION_BATCH_SIZE, limit: int | None = None) -> None:
    module = load_modern(root)
    model, make_dataset, provenance = module.model_and_provenance(root, "TolerantECG")
    model = model.to(device).eval().requires_grad_(False)
    for split in SPLITS:
        output = root / "results" / CAMPAIGN / "shared/masked_features" / task / split
        complete = output / "complete.json"
        if limit is None and complete.is_file():
            value = json.loads(complete.read_text(encoding="utf-8"))
            if value.get("state") == "complete":
                continue
        dataset_name = TASK_CONFIG[task][0]
        dataset = make_dataset(dataset_name, split)
        indices, ids, labels = merl_indices(root, task, split, dataset)
        baseline_manifest = verify_against_baseline(root, task, split, ids, labels)
        count = len(indices) if limit is None else min(limit, len(indices))
        output.mkdir(parents=True, exist_ok=True)
        path = output / ("features.npy.incoming" if limit is None else "canary.npy")
        features = open_memmap(path, mode="w+", dtype=np.float16,
                               shape=(count, LEADS, EMBED_DIM))
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            features[start:stop] = encode_masked(model, dataset.ecg[indices[start:stop]], device)
            if not np.isfinite(features[start:stop]).all():
                raise RuntimeError(f"non-finite masked features for {task}/{split}/{start}:{stop}")
            features.flush()
            atomic_json(output / "status.json", {"state": "extracting", "task": task,
                "split": split, "records_done": stop, "records_total": count})
        del features
        if limit is not None:
            continue
        os.replace(path, output / "features.npy")
        payload = {"state": "complete", "task": task, "split": split,
            "records": count, "shape": [count, LEADS, EMBED_DIM], "dtype": "float16",
            "view": "keep exactly one post-preprocessing lead; zero the other eleven",
            "record_ids_sha256": array_sha256(ids), "labels_sha256": array_sha256(labels),
            "baseline_record_ids_sha256": baseline_manifest["record_ids_sha256"],
            "checkpoint_sha256": provenance["checkpoint_sha256"], "batch_size": batch_size}
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

