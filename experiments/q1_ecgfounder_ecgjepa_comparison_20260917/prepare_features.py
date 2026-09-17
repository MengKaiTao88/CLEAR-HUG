#!/usr/bin/env python3
"""Extract audited frozen ECGFounder/ECG-JEPA representations on GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample


CAMPAIGN = "q1-ecgfounder-ecgjepa-six-task-three-fraction-seeds42-46-55-20260917"
SOURCE_CAMPAIGN = "q1-ecgfix-clocs-six-task-three-fraction-seeds42-46-55-20260916"
FOUNDER_COMMIT = "04edac702b61c91face519774ddcc0cd712fef23"
FOUNDER_SHA256 = "ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997"
JEPA_COMMIT = "d937ad2c2c8a1e22856ce7e4a23a30f84a71217c"
JEPA_SHA256 = "61334869f905a7d6de32bc573c60024eaf6efba7c35c0e45fc2ea7d52b6ff66e"
BASES = ("ptbxl", "cpsc2018", "csn")
BASE_ARRAYS = {
    "ptbxl": "ptbxl_ecg_500hz.npy",
    "cpsc2018": "cpsc2018_ecg.npy",
    "csn": "csn_ecg.npy",
}
TASKS = {
    "PTBXL_form": ("ptbxl", "ptbxl_form_metadata_final.csv", "ptbxl_form_labels.npy"),
    "PTBXL_super": ("ptbxl", "ptbxl_diagnostic_class_metadata_final.csv", "ptbxl_diagnostic_class_labels.npy"),
    "PTBXL_sub": ("ptbxl", "ptbxl_diagnostic_subclass_metadata_final.csv", "ptbxl_diagnostic_subclass_labels.npy"),
    "PTBXL_rhythm": ("ptbxl", "ptbxl_rhythm_metadata_final.csv", "ptbxl_rhythm_labels.npy"),
    "CPSC": ("cpsc2018", "cpsc2018_metadata_final.csv", "cpsc2018_labels.npy"),
    "CSN": ("csn", "csn_metadata_final.csv", "csn_labels.npy"),
}


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


def deployment_audit(root: Path, model: str) -> dict:
    if model == "ECGFounder":
        source = root / "external_models/ECGFounder"
        weight = root / "model_weights/ECGFounder/12_lead_ECGFounder.pth"
        expected_commit, expected_hash = FOUNDER_COMMIT, FOUNDER_SHA256
    else:
        source = root / "external_models/ECG_JEPA"
        weight = root / "model_weights/ECG_JEPA/multiblock_epoch100.pth"
        expected_commit, expected_hash = JEPA_COMMIT, JEPA_SHA256
    commit = (source / ".deployment-complete").read_text(encoding="utf-8").strip()
    weight_hash = sha256(weight)
    if commit != expected_commit or weight_hash != expected_hash:
        raise RuntimeError(f"{model} deployment identity mismatch: {commit=} {weight_hash=}")
    return {"model": model, "source_commit": commit, "checkpoint_sha256": weight_hash,
            "checkpoint": str(weight)}


def load_encoder(root: Path, model_name: str, device: torch.device):
    if model_name == "ECGFounder":
        source = root / "external_models/ECGFounder"
        sys.path.insert(0, str(source))
        from finetune_model import ft_12lead_ECGFounder
        model = ft_12lead_ECGFounder(
            device, root / "model_weights/ECGFounder/12_lead_ECGFounder.pth",
            n_classes=1, linear_prob=True)
        model.return_features = True

        def encode(x: torch.Tensor) -> torch.Tensor:
            _, features = model(x)
            return features
        expected_dim = 1024
    else:
        source = root / "external_models/ECG_JEPA"
        sys.path.insert(0, str(source))
        for module in ("models", "ecg_jepa", "pos_encoding"):
            sys.modules.pop(module, None)
        from models import load_encoder as load_jepa
        model, expected_dim = load_jepa(root / "model_weights/ECG_JEPA/multiblock_epoch100.pth")
        model = model.to(device)

        def encode(x: torch.Tensor) -> torch.Tensor:
            return model.representation(x)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != 0:
        raise RuntimeError("encoder is not fully frozen")
    return model, encode, expected_dim


def preprocess(batch: np.ndarray, model: str) -> np.ndarray:
    batch = np.asarray(batch, dtype=np.float32)
    if model == "ECGFounder":
        mean = batch.mean(axis=(1, 2), keepdims=True, dtype=np.float64).astype(np.float32)
        std = batch.std(axis=(1, 2), keepdims=True, dtype=np.float64).astype(np.float32)
        return (batch - mean) / np.maximum(std, 1e-8)
    # Official ECG-JEPA evaluation: I, II, V1..V6 and Fourier resampling to 250 Hz.
    batch = batch[:, [0, 1, 6, 7, 8, 9, 10, 11], :]
    return resample(batch, 2500, axis=2).astype(np.float32, copy=False)


def audit_tasks(root: Path) -> dict:
    source = root / "results" / SOURCE_CAMPAIGN / "shared"
    report = {}
    for task, (base, metadata_name, labels_name) in TASKS.items():
        metadata = pd.read_csv(source / "processed" / metadata_name)
        labels = np.load(source / "raw" / labels_name, mmap_mode="r")
        indices = metadata["ecg_index"].to_numpy(dtype=np.int64)
        label_columns = [name for name in metadata.columns if name.startswith("label_")]
        metadata_labels = metadata[label_columns].to_numpy(dtype=np.uint8)
        if not np.array_equal(metadata_labels, np.asarray(labels[indices], dtype=np.uint8)):
            raise RuntimeError(f"{task}: metadata/raw label order mismatch")
        split_sets = {}
        split_report = {}
        id_column = "ecg_id" if "ecg_id" in metadata else "record_id"
        for split in ("train", "val", "test"):
            mask = metadata["split"].eq(split).to_numpy()
            task_labels = metadata_labels[mask]
            clocs_labels = np.load(source / "embeddings" / task / split / "CLOCS_y.npy")
            if not np.array_equal(task_labels, clocs_labels):
                raise RuntimeError(f"{task}/{split}: labels differ from audited CLOCS order")
            ids = set(metadata.loc[mask, id_column].astype(str))
            split_sets[split] = ids
            split_report[split] = {"records": int(mask.sum()),
                                   "index_sha256": hashlib.sha256(indices[mask].tobytes()).hexdigest()}
        overlaps = {"train_val": len(split_sets["train"] & split_sets["val"]),
                    "train_test": len(split_sets["train"] & split_sets["test"]),
                    "val_test": len(split_sets["val"] & split_sets["test"])}
        if any(overlaps.values()):
            raise RuntimeError(f"{task}: record-level split leakage: {overlaps}")
        report[task] = {"base": base, "label_count": len(label_columns),
                        "splits": split_report, "record_id_overlaps": overlaps}
    return report


def extract(root: Path, model_name: str, bases: list[str], device: torch.device,
            batch_size: int) -> None:
    campaign = root / "results" / CAMPAIGN
    source = root / "results" / SOURCE_CAMPAIGN / "shared/raw"
    audit = deployment_audit(root, model_name)
    task_audit = audit_tasks(root)
    model, encode, expected_dim = load_encoder(root, model_name, device)
    worker_status = campaign / f"prepare-{model_name.lower()}-{'-'.join(bases)}-status.json"
    with torch.inference_mode():
        for base in bases:
            output = campaign / "features" / model_name / f"{base}.npy"
            done = output.with_suffix(".done.json")
            if done.is_file():
                value = json.loads(done.read_text(encoding="utf-8"))
                if value.get("checkpoint_sha256") == audit["checkpoint_sha256"]:
                    continue
            waves = np.load(source / BASE_ARRAYS[base], mmap_mode="r")
            output.parent.mkdir(parents=True, exist_ok=True)
            incoming = output.with_suffix(".npy.incoming")
            features = np.lib.format.open_memmap(incoming, mode="w+", dtype=np.float32,
                                                  shape=(len(waves), expected_dim))
            for start in range(0, len(waves), batch_size):
                stop = min(start + batch_size, len(waves))
                values = preprocess(waves[start:stop], model_name)
                tensor = torch.from_numpy(values).to(device, non_blocking=True)
                encoded = encode(tensor)
                if encoded.shape != (stop - start, expected_dim):
                    raise RuntimeError(f"unexpected {model_name} feature shape: {encoded.shape}")
                features[start:stop] = encoded.float().cpu().numpy()
                features.flush()
                atomic_json(worker_status, {"state": "extracting", "base": base,
                    "records_done": stop, "records_total": len(waves), "device": str(device), **audit})
            del features
            os.replace(incoming, output)
            payload = {"state": "complete", "base": base, "records": len(waves),
                       "embedding_dim": expected_dim, "device": str(device), **audit}
            atomic_json(done, payload)
    complete = {"state": "complete", "bases": bases, "device": str(device),
                "task_split_audit": task_audit, **audit}
    atomic_json(worker_status.with_name(worker_status.name.replace("-status", "-complete")), complete)
    atomic_json(worker_status, complete)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", choices=("ECGFounder", "ECG-JEPA"), required=True)
    parser.add_argument("--bases", nargs="+", choices=BASES, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("feature extraction requires CUDA")
    extract(args.root.resolve(), args.model, args.bases, device, args.batch_size)


if __name__ == "__main__":
    main()
