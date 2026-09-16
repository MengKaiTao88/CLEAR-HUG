#!/usr/bin/env python3
"""Audit ST-MEM identity/freeze/splits and run the CPSC seed-42 single-crop check."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import atomic_json, save_npy_atomic, sha256, state_dict
from protocol import STMEM_SHA256

AUDIT = "q1-stmem-cpsc-seed42-single-crop-audit-20260916"
SOURCE = "q1-heartlang-stmem-linear-probe-seeds42-46-55-20260914"


class SingleCropDataset(Dataset):
    def __init__(self, directory: Path, split: str, transform):
        self.data = np.load(directory / f"{split}_data.npy", mmap_mode="r")
        self.labels = np.load(directory / f"{split}_labels.npy", mmap_mode="r")
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        value = np.asarray(self.data[index], dtype=np.float32)
        if value.shape == (1000, 12):
            value = value.T
        return self.transform(value), np.array(self.labels[index], copy=True, dtype=np.float32)


def build(root: Path):
    repo = root / "external_models/ST-MEM"
    sys.path.insert(0, str(repo))
    from models.encoder.st_mem_vit import st_mem_vit_base
    import util.transforms as transforms
    checkpoint = root / "model_weights/ST-MEM/st_mem_vit_base_encoder.pth"
    if sha256(checkpoint) != STMEM_SHA256:
        raise RuntimeError("checkpoint SHA256 does not match the pinned official encoder")
    model = st_mem_vit_base(num_leads=12, num_classes=None, seq_len=2250, patch_size=75)
    payload = state_dict(torch.load(checkpoint, map_location="cpu"))
    payload = {key.removeprefix("module."): value for key, value in payload.items() if not key.startswith("head.")}
    incompatible = model.load_state_dict(payload, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"encoder checkpoint mismatch: {incompatible}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    pipeline = transforms.Compose([
        transforms.CenterCrop(crop_length=2250),
        transforms.HighpassFilter(fs=250, cutoff=0.67),
        transforms.LowpassFilter(fs=250, cutoff=40),
        transforms.Standardize(axis=(-1, -2)),
        transforms.ToTensor(),
    ])
    resample = transforms.Resample(target_fs=250)

    def transform(value):
        return pipeline(resample(value, 100))
    return model, checkpoint, transform, payload


def id_digest(values: np.ndarray) -> str:
    return hashlib.sha256("\n".join(map(str, values.tolist())).encode()).hexdigest()


def audit_splits(raw: Path) -> dict:
    ids = {split: np.load(raw / f"{split}_path.npy", allow_pickle=True) for split in ("train", "val", "test")}
    sets = {split: set(map(str, values.tolist())) for split, values in ids.items()}
    return {
        "record_counts": {split: len(values) for split, values in ids.items()},
        "source_record_id_sha256": {split: id_digest(values) for split, values in ids.items()},
        "source_record_id_overlap": {
            "train_val": len(sets["train"] & sets["val"]),
            "train_test": len(sets["train"] & sets["test"]),
            "val_test": len(sets["val"] & sets["test"]),
        },
        "pipeline_order": "record-level split files -> resample each record -> one fixed center crop",
    }


@torch.no_grad()
def extract(model, loader):
    model.eval(); rows = []; labels = []
    for signals, truth in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            rows.append(model.forward_encoding(signals.cuda(non_blocking=True).float()).float().cpu().numpy())
        labels.append(truth.numpy())
    return np.concatenate(rows).astype(np.float32), np.concatenate(labels).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("audit", "extract", "evaluate"), required=True)
    parser.add_argument("--split", choices=("train", "val", "test"))
    args = parser.parse_args()
    output = args.root / "results" / AUDIT
    raw = args.root / "campaign-inputs" / SOURCE / "raw/CPSC2018/data"
    model, checkpoint, transform, payload = build(args.root)
    if args.stage == "audit":
        head = torch.nn.Linear(768, 9)
        optimizer = torch.optim.AdamW(head.parameters(), lr=5e-3, weight_decay=0.05)
        encoder_parameters = {id(parameter) for parameter in model.parameters()}
        optimizer_parameters = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        report = {
            "status": "passed",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "expected_official_encoder_sha256": STMEM_SHA256,
            "checkpoint_top_level": list(torch.load(checkpoint, map_location="cpu").keys()),
            "encoder_state_tensors": len(payload),
            "checkpoint_contains_classifier_head": any(key.startswith("head.") for key in state_dict(torch.load(checkpoint, map_location="cpu"))),
            "encoder_total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "encoder_trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "linear_head_trainable_parameters": sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad),
            "optimizer_parameter_count": sum(parameter.numel() for group in optimizer.param_groups for parameter in group["params"]),
            "optimizer_contains_encoder_parameter": bool(encoder_parameters & optimizer_parameters),
            "splits": audit_splits(raw),
        }
        atomic_json(output / "identity-freeze-split-audit.json", report)
        print(json.dumps(report, indent=2)); return
    if args.stage == "evaluate":
        from sklearn.metrics import roc_auc_score
        feature_root = output / "features/stmem/cpsc2018/test"
        checkpoint_path = output / "checkpoint/checkpoint-best.pth"
        features = torch.from_numpy(np.asarray(np.load(feature_root / "features.npy"), dtype=np.float32))
        truth = np.asarray(np.load(feature_root / "labels.npy"), dtype=np.float32)
        payload = torch.load(checkpoint_path, map_location="cpu")
        head = torch.nn.Linear(768, 9); head.load_state_dict(payload["model"], strict=True); head.cuda().eval()
        logits = []
        with torch.inference_mode():
            for start in range(0, len(features), 1024):
                logits.append(head(features[start:start + 1024].cuda()).float().cpu().numpy())
        logits = np.concatenate(logits); probabilities = torch.sigmoid(torch.from_numpy(logits.astype(np.float64))).numpy()
        per_class = [float(roc_auc_score(truth[:, index], probabilities[:, index])) for index in range(truth.shape[1])]
        result = {"status": "complete", "metric_implementation": "sklearn.metrics.roc_auc_score",
                  "average": "macro", "macro_auroc": float(roc_auc_score(truth, probabilities, average="macro")),
                  "per_class_auroc": {str(index): value for index, value in enumerate(per_class)},
                  "records": len(truth), "classes": truth.shape[1], "checkpoint_sha256": sha256(checkpoint_path)}
        destination = output / "formal-test"; destination.mkdir(parents=True, exist_ok=True)
        incoming = destination / "test_predictions.npz.incoming"
        with incoming.open("wb") as handle:
            np.savez_compressed(handle, y_true=truth, logits=logits, probabilities=probabilities)
        os.replace(incoming, destination / "test_predictions.npz")
        atomic_json(destination / "independent-auroc.json", result); print(json.dumps(result, indent=2)); return
    if not args.split:
        raise RuntimeError("--split is required for extraction")
    if args.split == "test" and not (output / "checkpoint/training-complete.json").exists():
        raise RuntimeError("test extraction is gated until probe training completes")
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    model.cuda().eval()
    dataset = SingleCropDataset(raw, args.split, transform)
    features, labels = extract(model, DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True))
    destination = output / "features/stmem/cpsc2018" / args.split
    destination.mkdir(parents=True, exist_ok=True)
    save_npy_atomic(destination / "features.npy", features)
    save_npy_atomic(destination / "labels.npy", labels)
    manifest = {"status": "complete", "split": args.split, "records": len(labels),
                "features_sha256": sha256(destination / "features.npy"),
                "labels_sha256": sha256(destination / "labels.npy"),
                "encoder_sha256": STMEM_SHA256, "encoder_frozen": True,
                "crop": "one fixed center crop", "crop_length": 2250,
                "source_record_ids_sha256": audit_splits(raw)["source_record_id_sha256"][args.split]}
    atomic_json(destination / "feature-manifest.json", manifest); print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
