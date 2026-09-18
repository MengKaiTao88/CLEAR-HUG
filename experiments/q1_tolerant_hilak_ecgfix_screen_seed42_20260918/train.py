#!/usr/bin/env python3
"""Train HILA-K with the original ECG-FIX frozen-probe protocol; validation only."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import HILAK
from protocol import (BASELINE_CAMPAIGN, BATCH_SIZE, CAMPAIGN, FRACTION,
                      GATE_INITIAL_VALUE, LEARNING_RATE, MAX_EPOCHS,
                      MIN_EPOCHS, PATIENCE, SEED, TASKS, WEIGHT_DECAY)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = path.with_suffix(path.suffix + ".incoming")
    incoming.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(incoming, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def macro_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    aucs, auprcs = [], []
    for index in range(y_true.shape[1]):
        truth = y_true[:, index]
        if np.unique(truth).size == 2:
            aucs.append(roc_auc_score(truth, y_prob[:, index]))
        auprcs.append(average_precision_score(truth, y_prob[:, index]))
    if not aucs:
        raise RuntimeError("no validation classes contain both labels")
    return float(np.mean(aucs)), float(np.mean(auprcs))


class FeatureDataset(Dataset):
    def __init__(self, global_features, lead_features, labels,
                 mean: np.ndarray, std: np.ndarray) -> None:
        self.global_features = global_features
        self.lead_features = lead_features
        self.labels = labels
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        global_features = np.asarray(self.global_features[index], dtype=np.float32)
        lead_features = np.asarray(self.lead_features[index], dtype=np.float32)
        return ((global_features - self.mean) / self.std,
                (lead_features - self.mean[None, :]) / self.std[None, :],
                np.asarray(self.labels[index], dtype=np.float32))


def load_split(root: Path, task: str, split: str):
    baseline = root / "results" / BASELINE_CAMPAIGN / "shared/embeddings/TolerantECG" / task / split
    masked = root / "results" / CAMPAIGN / "shared/masked_features" / task / split
    manifest = json.loads((masked / "complete.json").read_text(encoding="utf-8"))
    if manifest.get("state") != "complete":
        raise RuntimeError(f"incomplete masked features: {masked}")
    global_features = np.load(baseline / "features.npy", mmap_mode="r")
    lead_features = np.load(masked / "features.npy", mmap_mode="r")
    labels = (np.load(baseline / "labels.npy", mmap_mode="r") > 0).astype(np.float32)
    if len(global_features) != len(lead_features) or len(labels) != len(lead_features):
        raise RuntimeError(f"feature length mismatch for {task}/{split}")
    return global_features, lead_features, labels, manifest


def baseline(root: Path, task: str):
    folder = root / "results" / BASELINE_CAMPAIGN / "TolerantECG" / f"seed-{SEED}" / task / f"{FRACTION:g}"
    result = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    # This is a trusted checkpoint produced locally by the first campaign. It
    # contains NumPy mean/std arrays in addition to tensor state.
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    if mean.shape != (768,) or std.shape != (768,) or np.any(std <= 0):
        raise RuntimeError(f"invalid baseline normalization for {task}")
    return checkpoint["state_dict"], mean, std, result


@torch.no_grad()
def predict(model: nn.Module, dataset: Dataset, device: torch.device) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0,
                        pin_memory=True)
    outputs = []
    model.eval()
    for global_features, lead_features, _ in loader:
        logits = model(global_features.to(device, non_blocking=True),
                       lead_features.to(device, non_blocking=True))
        outputs.append(torch.sigmoid(logits).cpu().numpy())
    probabilities = np.concatenate(outputs)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("validation probabilities contain non-finite values")
    return probabilities


def train(root: Path, task: str, device: torch.device) -> dict:
    output = root / "results" / CAMPAIGN / task
    complete = output / "complete.json"
    if complete.is_file():
        value = json.loads(complete.read_text(encoding="utf-8"))
        if value.get("state") == "complete":
            return value
    train_global, train_leads, train_labels, train_manifest = load_split(root, task, "train")
    val_global, val_leads, val_labels, val_manifest = load_split(root, task, "val")
    base_state, mean, std, base_result = baseline(root, task)
    seed_everything(SEED)
    train_dataset = FeatureDataset(train_global, train_leads, train_labels, mean, std)
    val_dataset = FeatureDataset(val_global, val_leads, val_labels, mean, std)
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED), num_workers=0,
                        pin_memory=True, drop_last=False)
    model = HILAK(train_labels.shape[1], base_state).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    history, best_state = [], None
    best_auc, best_epoch, stale = -float("inf"), 0, 0
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for global_features, lead_features, labels in loader:
            global_features = global_features.to(device, non_blocking=True)
            lead_features = lead_features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(global_features, lead_features), labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
        val_prob = predict(model, val_dataset, device)
        val_auc, val_auprc = macro_metrics(np.asarray(val_labels), val_prob)
        history.append({"epoch": epoch, "learning_rate": LEARNING_RATE,
                        "train_loss": loss_sum / len(train_dataset),
                        "val_macro_auroc": val_auc, "val_macro_auprc": val_auprc})
        if val_auc > best_auc + 1e-8:
            best_auc, best_epoch, stale = val_auc, epoch, 0
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        else:
            stale += 1
        atomic_json(output / "status.json", {"state": "training", "task": task,
            "epoch": epoch, "best_epoch": best_epoch,
            "best_val_macro_auroc": best_auc, "stale_epochs": stale})
        if epoch >= MIN_EPOCHS and stale >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    torch.save({"state_dict": best_state, "epoch": best_epoch}, output / "best.pt")
    atomic_json(output / "history.json", history)
    result = {"state": "complete", "protocol": "tolerant-hilak-ecgfix-validation-screen-v1",
        "task": task, "seed": SEED, "fraction": FRACTION,
        "train_records": len(train_dataset), "val_records": len(val_dataset),
        "best_epoch": best_epoch, "epochs_ran": len(history),
        "best_val_macro_auroc": best_auc,
        "best_val_macro_auprc": history[best_epoch - 1]["val_macro_auprc"],
        "baseline_val_macro_auroc": base_result["best_val_macro_auroc"],
        "val_auroc_delta": best_auc - base_result["best_val_macro_auroc"],
        "test_evaluations": 0, "test_data_loaded": False,
        "encoder_frozen": True, "baseline_head_frozen": True,
        "baseline_feature_standardization": True,
        "masked_feature_standardization": "same train mean/std as baseline global features",
        "residual_head_zero_initialized": True, "gate_initial_value": GATE_INITIAL_VALUE,
        "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS, "early_stopping_patience": PATIENCE,
        "scheduler": "none (fixed learning rate, matching first protocol)",
        "selection_metric": "validation_macro_auroc",
        "train_masked_manifest": train_manifest, "val_masked_manifest": val_manifest}
    atomic_json(complete, result)
    atomic_json(output / "status.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    print(json.dumps(train(args.root.resolve(), args.task, device), indent=2))


if __name__ == "__main__":
    main()
