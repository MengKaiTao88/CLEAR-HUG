#!/usr/bin/env python3
"""Run official MERL ResNet-18 linear probes for five remaining ECG tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb
from scipy.io import loadmat
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


CAMPAIGN = "q1-merl-five-task-three-seed-20260916"
SOURCE_COMMIT = "2a38649285e16eff75b69aeb64f2366b380c1a9e"
ENCODER_SHA256 = "38ba669c2cc319670c4172d8c292f123e86b8e7106b1a68bb7e10dd89f09daf5"
RATIOS = (1, 10, 100)

TASKS = {
    "superdiagnostic": {
        "kind": "ptbxl", "split_dir": "ptbxl/super_class", "prefix": "ptbxl_super_class",
        "array_dir": "PTBXL/superdiagnostic/data", "meta_columns": 6, "classes": 5,
        "counts": {"train": 17084, "val": 2146, "test": 2158},
    },
    "subdiagnostic": {
        "kind": "ptbxl", "split_dir": "ptbxl/sub_class", "prefix": "ptbxl_sub_class",
        "array_dir": "PTBXL/subdiagnostic/data", "meta_columns": 6, "classes": 23,
        "counts": {"train": 17084, "val": 2146, "test": 2158},
    },
    "form": {
        "kind": "ptbxl", "split_dir": "ptbxl/form", "prefix": "ptbxl_form",
        "array_dir": "PTBXL/form/data", "meta_columns": 6, "classes": 19,
        "counts": {"train": 7197, "val": 901, "test": 880},
    },
    "rhythm": {
        "kind": "ptbxl", "split_dir": "ptbxl/rhythm", "prefix": "ptbxl_rhythm",
        "array_dir": "PTBXL/rhythm/data", "meta_columns": 6, "classes": 12,
        "counts": {"train": 16832, "val": 2100, "test": 2098},
    },
    "csn": {
        "kind": "csn", "split_dir": "chapman", "prefix": "chapman",
        "array_dir": "CSN/data", "meta_columns": 3, "classes": 38,
        "counts": {"train": 16546, "val": 1860, "test": 4620},
    },
}

# Balanced by official full-training-set size: about 74.7k records per GPU
# across the five assigned task-seed groups before applying label fractions.
ASSIGNMENTS = {
    0: (("superdiagnostic", 42), ("subdiagnostic", 46), ("rhythm", 55), ("csn", 42), ("form", 46)),
    1: (("superdiagnostic", 46), ("subdiagnostic", 55), ("rhythm", 42), ("csn", 46), ("form", 55)),
    2: (("superdiagnostic", 55), ("subdiagnostic", 42), ("rhythm", 46), ("csn", 55), ("form", 42)),
}


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = path.with_suffix(path.suffix + ".incoming")
    incoming.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(incoming, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = torch.relu(self.bn1(self.conv1(value)))
        output = self.bn2(self.conv2(output))
        return torch.relu(output + self.shortcut(value))


class MERLResNet18(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(12, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._layer(64, 2, 1)
        self.layer2 = self._layer(128, 2, 2)
        self.layer3 = self._layer(256, 2, 2)
        self.layer4 = self._layer(512, 2, 2)
        self.linear = nn.Linear(512, classes)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def _layer(self, channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = []
        for current_stride in [stride] + [1] * (blocks - 1):
            layers.append(BasicBlock(self.in_channels, channels, current_stride))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = torch.relu(self.bn1(self.conv1(value)))
        output = self.layer4(self.layer3(self.layer2(self.layer1(output))))
        return self.linear(self.avgpool(output).view(output.size(0), -1))


class ECGDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, labels: tuple[str, ...], kind: str, root: Path):
        self.frame = frame.reset_index(drop=True)
        self.labels = self.frame[list(labels)].to_numpy(dtype=np.float32)
        self.kind = kind
        self.root = root
        self.paths = self.frame["filename_hr" if kind == "ptbxl" else "ecg_path"].astype(str).to_numpy()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        name = self.paths[index]
        if self.kind == "ptbxl":
            value = wfdb.rdsamp(str(self.root / name))[0].T[:, :5000]
        else:
            relative = name.removeprefix("/chapman/").lstrip("/")
            value = np.asarray(loadmat(self.root / relative)["val"], dtype=np.float32)[:, :5000]
        if value.shape != (12, 5000):
            raise RuntimeError(f"unexpected waveform shape {value.shape} for {name}")
        value = (value - value.min()) / (value.max() - value.min() + 1e-8)
        value[[4, 5]] = value[[5, 4]]
        return torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)), torch.from_numpy(self.labels[index])


def load_frames(root: Path, code_root: Path, task: str) -> tuple[dict[str, pd.DataFrame], tuple[str, ...], dict]:
    config = TASKS[task]
    array_root = root / "src/CLEAR-HUG/datasets/ecg_datasets" / config["array_dir"]
    split_root = code_root / "data_split" / config["split_dir"]
    frames, audit = {}, {"counts": {}, "array_order_and_labels_match": {}, "record_overlap": {}}
    labels = None
    for split in ("train", "val", "test"):
        frame = pd.read_csv(split_root / f"{config['prefix']}_{split}.csv")
        if len(frame) != config["counts"][split]:
            raise RuntimeError(f"{task} {split} count mismatch")
        current_labels = tuple(frame.columns[config["meta_columns"]:])
        if len(current_labels) != config["classes"]:
            raise RuntimeError(f"{task} label-count mismatch")
        labels = labels or current_labels
        if labels != current_labels:
            raise RuntimeError(f"{task} label order differs across splits")
        local_paths = np.load(array_root / f"{split}_path.npy", allow_pickle=True).astype(str)
        local_labels = np.load(array_root / f"{split}_labels.npy")
        official_paths = frame["filename_lr" if config["kind"] == "ptbxl" else "ecg_path"].astype(str).to_numpy()
        exact = np.array_equal(local_paths, official_paths) and np.array_equal(
            local_labels.astype(np.float32), frame[list(labels)].to_numpy(dtype=np.float32)
        )
        if not exact:
            raise RuntimeError(f"{task} local arrays do not exactly match official MERL {split}")
        frames[split] = frame
        audit["counts"][split] = len(frame)
        audit["array_order_and_labels_match"][split] = True
    identity = "patient_id" if config["kind"] == "ptbxl" else "ecg_path"
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(frames[left][identity].astype(str)) & set(frames[right][identity].astype(str))
        audit["record_overlap"][f"{left}_{right}"] = len(overlap)
        if overlap:
            raise RuntimeError(f"{task}: overlap between {left} and {right}")
    return frames, labels, audit


def build_model(root: Path, classes: int) -> tuple[nn.Module, dict]:
    checkpoint = root / "model_weights/MERL/res18_best_encoder.pth"
    if sha256(checkpoint) != ENCODER_SHA256:
        raise RuntimeError("official MERL encoder SHA256 mismatch")
    payload = torch.load(checkpoint, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    payload = {key.removeprefix("module."): value for key, value in payload.items()}
    released_head = {key: list(value.shape) for key, value in payload.items() if key.startswith("linear.")}
    payload = {key: value for key, value in payload.items() if not key.startswith("linear.")}
    model = MERLResNet18(classes)
    incompatible = model.load_state_dict(payload, strict=False)
    if set(incompatible.missing_keys) != {"linear.weight", "linear.bias"} or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.linear.parameters():
        parameter.requires_grad = True
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != 512 * classes + classes:
        raise RuntimeError("trainable parameter audit failed")
    return model.cuda(), {
        "trainable_parameters": trainable,
        "encoder_frozen": True,
        "removed_released_pretraining_head": released_head,
        "checkpoint_adapter": "removed only mismatched linear.* tensors",
    }


def scores(truth: np.ndarray, logits: np.ndarray, labels: tuple[str, ...]) -> dict:
    probability = torch.sigmoid(torch.from_numpy(logits.astype(np.float64))).numpy()
    return {
        "macro_auroc": float(roc_auc_score(truth, probability, average="macro")),
        "macro_auprc": float(average_precision_score(truth, probability, average="macro")),
        "per_class_auroc": {name: float(roc_auc_score(truth[:, i], probability[:, i])) for i, name in enumerate(labels)},
        "per_class_auprc": {name: float(average_precision_score(truth[:, i], probability[:, i])) for i, name in enumerate(labels)},
    }


@torch.inference_mode()
def infer(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    truths, logits = [], []
    for waveform, target in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(waveform.cuda(non_blocking=True))
        truths.append(target.numpy())
        logits.append(output.float().cpu().numpy())
    return np.concatenate(truths), np.concatenate(logits)


def train_unit(root: Path, code_root: Path, output: Path, task: str, seed: int, ratio: int) -> dict:
    unit = output / task / f"seed{seed}" / f"{ratio}pct"
    complete = unit / "complete.json"
    if complete.exists():
        return json.loads(complete.read_text(encoding="utf-8"))
    unit.mkdir(parents=True, exist_ok=True)
    frames, labels, split_audit = load_frames(root, code_root, task)
    train_frame = frames["train"]
    if ratio != 100:
        train_frame, _ = train_test_split(train_frame, train_size=ratio / 100, random_state=seed)
        train_frame = train_frame.reset_index(drop=True)
    config = TASKS[task]
    raw_root = (root / "data/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
                if config["kind"] == "ptbxl"
                else root / "src/CLEAR-HUG/datasets/dataset_preprocess/CSN")
    datasets = {
        "train": ECGDataset(train_frame, labels, config["kind"], raw_root),
        "val": ECGDataset(frames["val"], labels, config["kind"], raw_root),
        "test": ECGDataset(frames["test"], labels, config["kind"], raw_root),
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=16, shuffle=True, num_workers=6, pin_memory=True),
        "val": DataLoader(datasets["val"], batch_size=256, shuffle=False, num_workers=6, pin_memory=True),
        "test": DataLoader(datasets["test"], batch_size=256, shuffle=False, num_workers=6, pin_memory=True),
    }
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    model, model_audit = build_model(root, len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.1)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.BCEWithLogitsLoss()
    best, legacy_best, log = None, None, []
    for epoch in range(100):
        model.train()
        losses = []
        for waveform, target in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = criterion(model(waveform.cuda(non_blocking=True)), target.cuda(non_blocking=True))
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); scheduler.step()
            losses.append(float(loss.detach().cpu()))
        val_truth, val_logits = infer(model, loaders["val"])
        test_truth, test_logits = infer(model, loaders["test"])
        val_metric = scores(val_truth, val_logits, labels)
        test_metric = scores(test_truth, test_logits, labels)
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)),
               "learning_rate": optimizer.param_groups[0]["lr"],
               "validation": val_metric, "released_code_test_monitor": test_metric}
        log.append(row)
        if legacy_best is None or test_metric["macro_auroc"] > legacy_best["macro_auroc"]:
            legacy_best = {"epoch": epoch + 1, **test_metric}
        if best is None or val_metric["macro_auroc"] > best["validation"]["macro_auroc"]:
            best = {"epoch": epoch + 1, "validation": val_metric}
            torch.save({"model": model.state_dict(), **best}, unit / "checkpoint-best-validation.pth")
        atomic_json(unit / "status.json", {"state": "training", "task": task, "seed": seed,
                                            "ratio": ratio, **row})
        print(json.dumps({"task": task, "seed": seed, "ratio": ratio, **row}), flush=True)
        scheduler.step()
    payload = torch.load(unit / "checkpoint-best-validation.pth", map_location="cpu")
    model.load_state_dict(payload["model"])
    formal_truth, formal_logits = infer(model, loaders["test"])
    formal_probability = torch.sigmoid(torch.from_numpy(formal_logits.astype(np.float64))).numpy()
    np.savez_compressed(unit / "test_predictions.npz", y_true=formal_truth, y_prob=formal_probability,
                        record_ids=datasets["test"].paths, labels=np.asarray(labels))
    atomic_json(unit / "training-log.json", {"epochs": log})
    result = {
        "status": "complete", "task": task, "seed": seed, "fraction": f"{ratio}pct",
        "train_records": len(train_frame), "labels": labels, "split_audit": split_audit,
        "model_audit": model_audit, "best_validation": best,
        "validation_selected_test": scores(formal_truth, formal_logits, labels),
        "released_code_max_test_over_epochs": legacy_best,
        "checkpoint_sha256": sha256(unit / "checkpoint-best-validation.pth"),
        "test_selection_warning": "use validation_selected_test for fair comparison",
    }
    atomic_json(complete, result)
    atomic_json(unit / "status.json", {"state": "complete", **result})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True, choices=(0, 1, 2))
    args = parser.parse_args()
    root = args.root.resolve()
    code_root = root / "src/CLEAR-HUG/experiments/q1_merl_five_task_three_seed_20260916"
    output = root / "results" / CAMPAIGN
    status = output / f"worker-{args.gpu}-status.json"
    units = [(task, seed, ratio) for task, seed in ASSIGNMENTS[args.gpu] for ratio in RATIOS]
    completed = 0
    try:
        for task, seed, ratio in units:
            atomic_json(status, {"state": "running", "gpu": args.gpu, "task": task, "seed": seed,
                                 "ratio": ratio, "completed_units": completed, "total_units": len(units)})
            train_unit(root, code_root, output, task, seed, ratio)
            completed += 1
        atomic_json(status, {"state": "complete", "gpu": args.gpu,
                             "completed_units": completed, "total_units": len(units)})
    except Exception as error:
        atomic_json(status, {"state": "failed", "gpu": args.gpu, "completed_units": completed,
                             "error": repr(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
