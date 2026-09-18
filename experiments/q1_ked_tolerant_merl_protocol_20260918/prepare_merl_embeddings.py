#!/usr/bin/env python3
"""Reorder audited KED/TolerantECG embeddings to the exact official MERL splits."""
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
import torch.nn.functional as functional
import wfdb
from numpy.lib.format import open_memmap

from protocol import CAMPAIGN, MODELS, SOURCE_CAMPAIGN, SOURCE_EXPERIMENT, SPLITS, TASKS


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = path.with_suffix(path.suffix + ".incoming")
    incoming.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(incoming, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def load_modern_module(root: Path):
    source = root / "src/CLEAR-HUG/experiments" / SOURCE_EXPERIMENT / "prepare_embeddings.py"
    spec = importlib.util.spec_from_file_location("modern_prepare", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source


def canonical_key(kind: str, value) -> str:
    if kind == "ptbxl":
        return str(int(value))
    if kind == "csn":
        return Path(str(value)).stem
    return str(value)


def csv_key(kind: str, frame: pd.DataFrame) -> np.ndarray:
    if kind == "ptbxl":
        return frame["ecg_id"].map(lambda value: str(int(value))).to_numpy(dtype=str)
    if kind == "csn":
        return frame["ecg_path"].map(lambda value: Path(str(value)).stem).to_numpy(dtype=str)
    return frame["filename"].astype(str).to_numpy()


def metadata_key(kind: str, frame: pd.DataFrame) -> np.ndarray:
    column = "ecg_id" if kind == "ptbxl" else "record_id"
    return frame[column].map(lambda value: canonical_key(kind, value)).to_numpy(dtype=str)


def source_feature_table(root: Path, model: str, dataset: str, make_dataset) -> tuple[np.ndarray, dict]:
    source = root / "results" / SOURCE_CAMPAIGN / "shared/embeddings" / model / dataset
    chunks, indices, audits = [], [], []
    for split in SPLITS:
        folder = source / split
        meta = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
        features = np.load(folder / "features.npy", mmap_mode="r")
        ds = make_dataset(dataset, split)
        split_indices = np.asarray(ds.indices, dtype=np.int64)
        if len(features) != len(split_indices) or int(meta["records"]) != len(split_indices):
            raise RuntimeError(f"source feature/index mismatch: {model}/{dataset}/{split}")
        chunks.append(np.asarray(features, dtype=np.float32))
        indices.append(split_indices)
        audits.append({"split": split, "records": len(split_indices),
                       "source_indices_sha256": array_sha256(split_indices),
                       "source_manifest_sha256": sha256(folder / "complete.json")})
    all_indices = np.concatenate(indices)
    if len(np.unique(all_indices)) != len(all_indices):
        raise RuntimeError(f"duplicate source indices for {model}/{dataset}")
    max_index = int(all_indices.max())
    table = np.full((max_index + 1, chunks[0].shape[1]), np.nan, dtype=np.float32)
    for split_indices, features in zip(indices, chunks):
        table[split_indices] = features
    return table, {"source_splits": audits, "covered_indices": len(all_indices),
                   "feature_dim": int(table.shape[1])}


def recover_missing_records(root: Path, module, model_name: str, kind: str,
                            ecg_indices: np.ndarray, feature_table: np.ndarray,
                            record_names: list[str] | None = None) -> tuple[np.ndarray, dict]:
    missing = np.unique(ecg_indices[np.isnan(feature_table[ecg_indices]).any(axis=1)])
    if not len(missing):
        return feature_table, {"recovered_indices": [], "recovered_records": 0}
    model, _, provenance = module.model_and_provenance(root, model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    if kind == "cpsc":
        raw_root = root / "data/ecg-fix-physionet/challenge-2020/1.0.2/training/cpsc_2018"
        names = [f"A{index + 1:04d}" for index in missing.tolist()]
        paths = [raw_root / name for name in names]
    elif kind == "csn":
        raw_root = root / "src/CLEAR-HUG/datasets/dataset_preprocess/CSN/WFDBRecords"
        if record_names is None:
            raise RuntimeError("CSN recovery requires record names")
        names_by_index = dict(zip(ecg_indices.tolist(), record_names))
        names = [names_by_index[index] for index in missing.tolist()]
        paths = []
        for name in names:
            matches = list(raw_root.glob(f"**/{name}.hea"))
            if len(matches) != 1:
                raise RuntimeError(f"cannot uniquely resolve CSN record {name}: {matches}")
            paths.append(matches[0].with_suffix(""))
    else:
        raise RuntimeError(f"recovery is unsupported for {kind}")
    tensors = []
    for name, path in zip(names, paths):
        signal, _ = wfdb.rdsamp(str(path))
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        value = torch.from_numpy(np.ascontiguousarray(signal.T, dtype=np.float32))
        if value.shape[1] > 5000:
            start = (value.shape[1] - 5000) // 2
            value = value[:, start:start + 5000]
        elif value.shape[1] < 5000:
            pad = 5000 - value.shape[1]
            value = (functional.pad(value, (0, pad)) if kind == "csn" else
                     functional.pad(value, (pad // 2, pad - pad // 2)))
        if tuple(value.shape) != (12, 5000):
            raise RuntimeError(f"cannot recover CPSC record {name}: {tuple(value.shape)}")
        tensors.append(value)
    with torch.inference_mode():
        recovered = module.get_features(model_name, model, torch.stack(tensors).to(device))
    recovered = recovered.float().cpu().numpy()
    if recovered.shape != (len(missing), feature_table.shape[1]):
        raise RuntimeError(f"unexpected recovered CPSC feature shape: {recovered.shape}")
    feature_table[missing] = recovered
    return feature_table, {
        "recovered_indices": missing.tolist(),
        "recovered_record_ids": names,
        "recovered_records": len(missing),
        "recovered_features_sha256": array_sha256(recovered),
        "recovery_checkpoint_sha256": provenance["checkpoint_sha256"],
        "recovery_preprocessing": (
            "non-finite values replaced by zero; ECG-FIX center crop and center pad for CPSC; "
            "center crop and right pad for CSN"
        ),
    }


def prepare_model(root: Path, model: str) -> None:
    module, source_file = load_modern_module(root)
    _, _, make_dataset = module.configure_ecgfix(root)
    processed = root / "results" / module.SOURCE_DATA_CAMPAIGN / "shared/processed"
    split_root = root / "src/CLEAR-HUG/experiments/q1_ked_tolerant_merl_protocol_20260918/data_split"
    campaign = root / "results" / CAMPAIGN
    complete_count = 0
    tables: dict[str, tuple[np.ndarray, dict]] = {}
    for task, config in TASKS.items():
        dataset = config["source_dataset"]
        if dataset not in tables:
            tables[dataset] = source_feature_table(root, model, dataset, make_dataset)
        feature_table, source_audit = tables[dataset]
        metadata_path = processed / config["metadata"]
        metadata = pd.read_csv(metadata_path)
        keys = metadata_key(config["kind"], metadata)
        if len(set(keys)) != len(keys):
            raise RuntimeError(f"duplicate metadata keys in {metadata_path}")
        index_by_key = dict(zip(keys, metadata["ecg_index"].astype(np.int64)))
        canonical_labels = None
        split_key_sets = {}
        for split in SPLITS:
            output = campaign / "shared/merl_embeddings" / model / task / split
            done = output / "complete.json"
            csv_path = split_root / config["split_dir"] / f"{config['prefix']}_{split}.csv"
            frame = pd.read_csv(csv_path)
            if len(frame) != config["counts"][split]:
                raise RuntimeError(f"MERL count mismatch: {task}/{split}={len(frame)}")
            labels_here = tuple(frame.columns[config["meta_columns"]:])
            if len(labels_here) != config["classes"]:
                raise RuntimeError(f"MERL label count mismatch: {task}/{split}")
            if canonical_labels is None:
                canonical_labels = labels_here
            elif set(canonical_labels) != set(labels_here):
                raise RuntimeError(f"MERL label set mismatch: {task}/{split}")
            record_ids = csv_key(config["kind"], frame)
            if len(set(record_ids)) != len(record_ids):
                raise RuntimeError(f"duplicate MERL records: {task}/{split}")
            if config["kind"] == "cpsc":
                ecg_indices = frame["ecg_id"].to_numpy(dtype=np.int64) - 1
                feature_table, recovery_audit = recover_missing_records(
                    root, module, model, "cpsc", ecg_indices, feature_table)
            elif config["kind"] == "csn":
                raw_root = root / "src/CLEAR-HUG/datasets/dataset_preprocess/CSN/WFDBRecords"
                full_names = [path.stem for path in sorted(raw_root.glob("**/*.hea"))]
                if len(full_names) != 23026 or len(set(full_names)) != 23026:
                    raise RuntimeError(f"unexpected CSN raw inventory: {len(full_names)}")
                ecg_indices = np.arange(len(full_names), dtype=np.int64)
                feature_table, recovery_audit = recover_missing_records(
                    root, module, model, "csn", ecg_indices, feature_table, full_names)
                raw_index_by_name = {name: index for index, name in enumerate(full_names)}
                ecg_indices = np.asarray([raw_index_by_name[value] for value in record_ids], dtype=np.int64)
            else:
                missing = [value for value in record_ids if value not in index_by_key]
                if missing:
                    raise RuntimeError(f"unmapped MERL records: {task}/{split}: {missing[:5]}")
                ecg_indices = np.asarray([index_by_key[value] for value in record_ids], dtype=np.int64)
                recovery_audit = {"recovered_indices": [], "recovered_records": 0}
            if int(ecg_indices.max()) >= len(feature_table) or np.isnan(feature_table[ecg_indices]).any():
                raise RuntimeError(f"missing source embeddings: {model}/{task}/{split}")
            labels = frame[list(canonical_labels)].to_numpy(dtype=np.float32)
            output.mkdir(parents=True, exist_ok=True)
            features_out = open_memmap(output / "features.npy.incoming", mode="w+", dtype=np.float32,
                                       shape=(len(frame), feature_table.shape[1]))
            features_out[:] = feature_table[ecg_indices]
            features_out.flush(); del features_out
            np.save(output / "labels.npy.incoming", labels)
            os.replace(output / "labels.npy.incoming.npy", output / "labels.npy")
            np.save(output / "record_ids.npy.incoming", record_ids.astype(f"<U{max(map(len, record_ids))}"))
            os.replace(output / "record_ids.npy.incoming.npy", output / "record_ids.npy")
            os.replace(output / "features.npy.incoming", output / "features.npy")
            split_key_sets[split] = set(record_ids.tolist())
            payload = {
                "state": "complete", "model": model, "task": task, "split": split,
                "records": len(frame), "classes": config["classes"],
                "feature_shape": [len(frame), int(feature_table.shape[1])],
                "label_order": list(canonical_labels), "official_merl_csv": str(csv_path),
                "official_merl_csv_sha256": sha256(csv_path),
                "metadata_sha256": sha256(metadata_path),
                "record_ids_sha256": array_sha256(record_ids.astype(f"<U{max(map(len, record_ids))}")),
                "ecg_indices_sha256": array_sha256(ecg_indices),
                "labels_sha256": array_sha256(labels),
                "source_campaign": SOURCE_CAMPAIGN,
                "source_prepare_sha256": sha256(source_file), **source_audit, **recovery_audit,
            }
            atomic_json(done, payload); complete_count += 1
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = split_key_sets[left] & split_key_sets[right]
            if overlap:
                raise RuntimeError(f"MERL record overlap: {task}/{left}-{right}: {len(overlap)}")
    atomic_json(campaign / f"{model}-merl-embeddings-complete.json", {
        "state": "complete", "model": model, "completed_splits": complete_count,
        "expected_splits": len(TASKS) * len(SPLITS), "source_campaign": SOURCE_CAMPAIGN,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()
    prepare_model(args.root.resolve(), args.model)


if __name__ == "__main__":
    main()
