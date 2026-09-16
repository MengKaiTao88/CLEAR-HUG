#!/usr/bin/env python3
"""Reproduce the released MERL ResNet-18 linear probe on CPSC2018.

The model, split, preprocessing and optimizer follow MERL-ICML2024 commit
2a38649285e16eff75b69aeb64f2366b380c1a9e.  In addition to the released
code-compatible epoch metrics, this runner persists a validation-AUROC-selected
checkpoint and a clearly separated final result for each label fraction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


CAMPAIGN = "q1-merl-cpsc-official-20260916"
SOURCE_COMMIT = "2a38649285e16eff75b69aeb64f2366b380c1a9e"
ENCODER_SHA256 = "38ba669c2cc319670c4172d8c292f123e86b8e7106b1a68bb7e10dd89f09daf5"
LABELS = ("AFIB", "VPC", "NORM", "1AVB", "CRBBB", "STE", "PAC", "CLBBB", "STD")
SPLIT_COUNTS = {"train": 4950, "val": 551, "test": 1376}
RATIOS = (1, 10, 100)


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
    expansion = 1

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
        output += self.shortcut(value)
        return torch.relu(output)


class MERLResNet18(nn.Module):
    """Exact released MERL finetune/models/resnet1d.py ResNet-18."""

    def __init__(self, classes: int = 9):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(12, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.linear = nn.Linear(512, classes)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, channels: int, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for current_stride in strides:
            layers.append(BasicBlock(self.in_channels, channels, current_stride))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = torch.relu(self.bn1(self.conv1(value)))
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.avgpool(output).view(output.size(0), -1)
        return self.linear(output)


class CPSC(Dataset):
    def __init__(self, frame: pd.DataFrame, raw_root: Path):
        self.frame = frame.reset_index(drop=True)
        self.names = self.frame["filename"].astype(str).to_numpy()
        self.labels = self.frame[list(LABELS)].to_numpy(dtype=np.float32)
        self.raw_root = raw_root

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        # Matches the released MERL ECGDataset implementation.  The challenge
        # WFDB headers reference .mat signal files, so wfdb.rdsamp remains the
        # canonical physical-unit reader.
        value = wfdb.rdsamp(str(self.raw_root / self.names[index]))[0].T
        value = value[:, :2500]
        if value.shape[1] != 2500:
            raise RuntimeError(f"{self.names[index]} has only {value.shape[1]} samples")
        value = np.pad(value, ((0, 0), (0, 2500)), mode="constant")[:, :5000]
        value = (value - value.min()) / (value.max() - value.min() + 1e-8)
        value[[4, 5]] = value[[5, 4]]
        return torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)), torch.from_numpy(self.labels[index])


def metrics(truth: np.ndarray, logits: np.ndarray) -> dict[str, object]:
    probability = torch.sigmoid(torch.from_numpy(logits.astype(np.float64))).numpy()
    return {
        "macro_auroc": float(roc_auc_score(truth, probability, average="macro")),
        "macro_auprc": float(average_precision_score(truth, probability, average="macro")),
        "per_class_auroc": {name: float(roc_auc_score(truth[:, i], probability[:, i])) for i, name in enumerate(LABELS)},
        "per_class_auprc": {name: float(average_precision_score(truth[:, i], probability[:, i])) for i, name in enumerate(LABELS)},
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


def load_frames(split_root: Path, array_root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    frames, audit = {}, {"counts": {}, "array_order_and_labels_match": {}, "patient_overlap": {}}
    for split in ("train", "val", "test"):
        frame = pd.read_csv(split_root / f"icbeb_{split}.csv")
        if len(frame) != SPLIT_COUNTS[split]:
            raise RuntimeError(f"official {split} count mismatch: {len(frame)}")
        paths = np.load(array_root / f"{split}_path.npy", allow_pickle=True).astype(str)
        labels = np.load(array_root / f"{split}_labels.npy")
        exact = np.array_equal(paths, frame["filename"].astype(str).to_numpy()) and np.array_equal(
            labels.astype(np.float32), frame[list(LABELS)].to_numpy(dtype=np.float32)
        )
        if not exact:
            raise RuntimeError(f"local CPSC arrays do not exactly match official MERL {split} split")
        frames[split] = frame
        audit["counts"][split] = len(frame)
        audit["array_order_and_labels_match"][split] = True
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(frames[left]["patient_id"].astype(str)) & set(frames[right]["patient_id"].astype(str))
        audit["patient_overlap"][f"{left}_{right}"] = len(overlap)
        if overlap:
            raise RuntimeError(f"patient leakage between {left} and {right}: {len(overlap)}")
    return frames, audit


def build_model(checkpoint: Path) -> tuple[nn.Module, dict[str, object]]:
    if sha256(checkpoint) != ENCODER_SHA256:
        raise RuntimeError("official MERL ResNet-18 encoder SHA256 mismatch")
    model = MERLResNet18(9)
    payload = torch.load(checkpoint, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    payload = {key.removeprefix("module."): value for key, value in payload.items()}
    # Despite its released "encoder" filename, the ResNet checkpoint contains
    # a 10-class pretraining head.  The public downstream script asks
    # load_state_dict(strict=False) to ignore it, but PyTorch still raises on
    # shape mismatch against CPSC's 9-class head.  Removing only linear.* is
    # the minimal adapter that realizes the source code's stated intention.
    released_head = {key: list(value.shape) for key, value in payload.items() if key.startswith("linear.")}
    payload = {key: value for key, value in payload.items() if not key.startswith("linear.")}
    incompatible = model.load_state_dict(payload, strict=False)
    allowed_missing = {"linear.weight", "linear.bias"}
    if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint/model mismatch: {incompatible}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.linear.parameters():
        parameter.requires_grad = True
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != 512 * 9 + 9:
        raise RuntimeError(f"unexpected trainable parameter count: {trainable}")
    return model.cuda(), {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "removed_released_pretraining_head": released_head,
        "checkpoint_adapter": "removed only linear.* because released encoder carries a mismatched 10-class head",
        "trainable_parameters": trainable,
        "encoder_parameters_in_optimizer_with_grad": 0,
    }


def subset_frame(frame: pd.DataFrame, ratio: int) -> pd.DataFrame:
    if ratio == 100:
        return frame.reset_index(drop=True)
    selected, _ = train_test_split(frame, train_size=ratio / 100, random_state=42)
    return selected.reset_index(drop=True)


def train_ratio(root: Path, frames: dict[str, pd.DataFrame], ratio: int, output: Path) -> dict[str, object]:
    ratio_output = output / f"{ratio}pct"
    complete = ratio_output / "complete.json"
    if complete.exists():
        return json.loads(complete.read_text(encoding="utf-8"))
    ratio_output.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True

    raw_root = root / "src/CLEAR-HUG/datasets/dataset_preprocess/CPSC2018/kaggle/Training_WFDB"
    train_frame = subset_frame(frames["train"], ratio)
    datasets = {
        "train": CPSC(train_frame, raw_root),
        "val": CPSC(frames["val"], raw_root),
        "test": CPSC(frames["test"], raw_root),
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=16, shuffle=True, num_workers=8, pin_memory=True),
        "val": DataLoader(datasets["val"], batch_size=256, shuffle=False, num_workers=8, pin_memory=True),
        "test": DataLoader(datasets["test"], batch_size=256, shuffle=False, num_workers=8, pin_memory=True),
    }
    checkpoint = root / "model_weights/MERL/res18_best_encoder.pth"
    model, model_audit = build_model(checkpoint)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.1)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()
    best, rows, legacy_best_test = None, [], None

    for epoch in range(100):
        # The released linear-probe source calls train(), so frozen BatchNorm
        # running statistics update.  This behavior is intentionally retained.
        model.train()
        losses = []
        for waveform, target in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                output_batch = model(waveform.cuda(non_blocking=True))
                loss = criterion(output_batch, target.cuda(non_blocking=True))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # released code steps once per batch
            losses.append(float(loss.detach().cpu()))
        val_truth, val_logits = infer(model, loaders["val"])
        test_truth, test_logits = infer(model, loaders["test"])
        val_metric = metrics(val_truth, val_logits)
        test_metric = metrics(test_truth, test_logits)
        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation": val_metric,
            "released_code_test_monitor": test_metric,
        }
        rows.append(row)
        if legacy_best_test is None or test_metric["macro_auroc"] > legacy_best_test["macro_auroc"]:
            legacy_best_test = {"epoch": epoch + 1, **test_metric}
        if best is None or val_metric["macro_auroc"] > best["validation"]["macro_auroc"]:
            best = {"epoch": epoch + 1, "validation": val_metric}
            torch.save(
                {"model": model.state_dict(), "epoch": epoch + 1, "validation": val_metric},
                ratio_output / "checkpoint-best-validation.pth",
            )
        atomic_json(ratio_output / "status.json", {"state": "training", "ratio": ratio, **row})
        print(json.dumps({"ratio": ratio, **row}), flush=True)
        scheduler.step()  # released code also steps once per epoch

    payload = torch.load(ratio_output / "checkpoint-best-validation.pth", map_location="cpu")
    model.load_state_dict(payload["model"])
    formal_truth, formal_logits = infer(model, loaders["test"])
    formal_probability = torch.sigmoid(torch.from_numpy(formal_logits.astype(np.float64))).numpy()
    np.savez_compressed(
        ratio_output / "test_predictions.npz",
        y_true=formal_truth,
        y_prob=formal_probability,
        record_ids=frames["test"]["filename"].astype(str).to_numpy(),
    )
    atomic_json(ratio_output / "training-log.json", {"epochs": rows})
    result = {
        "status": "complete",
        "fraction": f"{ratio}pct",
        "seed": 42,
        "train_records": len(train_frame),
        "model_audit": model_audit,
        "best_validation": best,
        "validation_selected_test": metrics(formal_truth, formal_logits),
        "released_code_max_test_over_epochs": legacy_best_test,
        "test_selection_warning": "released code evaluates test every epoch; use validation_selected_test for fair comparison",
        "checkpoint_sha256": sha256(ratio_output / "checkpoint-best-validation.pth"),
    }
    atomic_json(complete, result)
    atomic_json(ratio_output / "status.json", {"state": "complete", **result})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "results" / CAMPAIGN
    output.mkdir(parents=True, exist_ok=True)
    status = output / "queue-status.json"
    try:
        atomic_json(status, {"state": "preflight"})
        code_root = root / "src/CLEAR-HUG/experiments/q1_merl_cpsc_official_20260916"
        frames, split_audit = load_frames(
            code_root / "data_split",
            root / "src/CLEAR-HUG/datasets/ecg_datasets/CPSC2018/data",
        )
        manifest = {
            "campaign": CAMPAIGN,
            "source_repository": "https://github.com/cheliu-computation/MERL-ICML2024",
            "source_commit": SOURCE_COMMIT,
            "encoder": "official released MERL ResNet-18 encoder",
            "encoder_sha256": ENCODER_SHA256,
            "split_audit": split_audit,
            "ratios": list(RATIOS),
            "seed_behavior": {"torch": 42, "python": 0, "numpy": 0, "fraction_split_random_state": 42},
            "optimizer": "Adam(lr=1e-3, weight_decay=1e-4)",
            "batch_size": 16,
            "epochs": 100,
            "scheduler": "MultiStepLR([40],0.1), released per-batch plus per-epoch stepping retained",
            "preprocessing": "first 2500 samples, zero-pad to 5000, record min-max, swap aVL/aVF",
        }
        atomic_json(output / "manifest.json", manifest)
        results = []
        for ratio in RATIOS:
            atomic_json(status, {"state": "running", "ratio": ratio, "completed_ratios": len(results)})
            results.append(train_ratio(root, frames, ratio, output))
        atomic_json(output / "summary.json", {"status": "complete", "results": results})
        atomic_json(status, {"state": "complete", "completed_ratios": len(results)})
    except Exception as error:
        atomic_json(status, {"state": "failed", "error": repr(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
