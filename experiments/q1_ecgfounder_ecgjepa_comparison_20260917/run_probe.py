#!/usr/bin/env python3
"""Shared validation-selected PyTorch linear probe for frozen comparison features."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from prepare_features import CAMPAIGN, SOURCE_CAMPAIGN, TASKS


MODELS = ("ECGFounder", "ECG-JEPA")
FRACTIONS = (0.01, 0.1, 1.0)
SEEDS = (42, 46, 55)
MAX_EPOCHS, MIN_EPOCHS, PATIENCE = 100, 10, 12
BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY = 256, 1e-3, 1e-4


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = path.with_suffix(path.suffix + ".incoming")
    incoming.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(incoming, path)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def macro_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float, list[dict]]:
    aucs, auprcs, rows = [], [], []
    for index in range(y_true.shape[1]):
        truth = y_true[:, index]
        auc = None if np.unique(truth).size < 2 else float(roc_auc_score(truth, y_prob[:, index]))
        if auc is not None: aucs.append(auc)
        auprc = float(average_precision_score(truth, y_prob[:, index]))
        auprcs.append(auprc)
        rows.append({"class_index": index, "auroc": auc, "auprc": auprc,
                     "positives": int(truth.sum()), "total": len(truth)})
    return float(np.mean(aucs)), float(np.mean(auprcs)), rows


@torch.no_grad()
def predict(model: nn.Module, values: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval(); outputs = []
    for start in range(0, len(values), 4096):
        x = torch.from_numpy(values[start:start + 4096]).to(device)
        outputs.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(outputs)


def load_task(root: Path, model_name: str, task: str):
    base, metadata_name, labels_name = TASKS[task]
    source = root / "results" / SOURCE_CAMPAIGN / "shared"
    metadata = pd.read_csv(source / "processed" / metadata_name)
    raw_labels = np.load(source / "raw" / labels_name, mmap_mode="r")
    base_features = np.load(root / "results" / CAMPAIGN / "features" / model_name / f"{base}.npy",
                            mmap_mode="r")
    indices = metadata["ecg_index"].to_numpy(dtype=np.int64)
    label_columns = [name for name in metadata.columns if name.startswith("label_")]
    labels = np.asarray(raw_labels[indices], dtype=np.float32)
    values = {}
    for split in ("train", "val", "test"):
        mask = metadata["split"].eq(split).to_numpy()
        values[split] = (np.asarray(base_features[indices[mask]], dtype=np.float32), labels[mask], indices[mask])
    return values


def train_unit(root: Path, model_name: str, task: str, fraction: float, seed: int,
               device: torch.device) -> dict:
    output = root / "results" / CAMPAIGN / "results" / model_name / f"seed-{seed}" / task / f"{fraction:g}"
    done = output / "complete.json"
    if done.is_file():
        value = json.loads(done.read_text(encoding="utf-8"))
        if value.get("state") == "complete" and value.get("protocol") == "pytorch-linear-v1":
            return value
    seed_everything(seed)
    splits = load_task(root, model_name, task)
    x_train, y_train, train_record_indices = splits["train"]
    x_val, y_val, _ = splits["val"]
    x_test, y_test, test_record_indices = splits["test"]
    generator = torch.Generator().manual_seed(seed)
    count = max(1, int(len(x_train) * fraction))
    selection = torch.randperm(len(x_train), generator=generator)[:count].numpy()
    train_x, train_y = x_train[selection], y_train[selection]
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=0, dtype=np.float64).astype(np.float32); std[std < 1e-6] = 1.0
    train_x = (train_x - mean) / std
    x_val = (x_val - mean) / std; x_test = (x_test - mean) / std
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
                        batch_size=BATCH_SIZE, shuffle=True, generator=generator,
                        num_workers=0, pin_memory=True)
    model = nn.Linear(train_x.shape[1], train_y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    best_auc, best_epoch, best_state, stale, history = -float("inf"), 0, None, 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); loss_sum = 0.0; examples = 0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels); loss.backward(); optimizer.step()
            loss_sum += float(loss.detach()) * len(features); examples += len(features)
        val_prob = predict(model, x_val, device)
        val_auc, val_auprc, _ = macro_metrics(y_val, val_prob)
        history.append({"epoch": epoch, "train_loss": loss_sum / examples,
                        "val_macro_auroc": val_auc, "val_macro_auprc": val_auprc})
        atomic_json(output / "status.json", {"state": "training", "model": model_name,
            "task": task, "fraction": fraction, "seed": seed, "epoch": epoch,
            "best_epoch": best_epoch, "best_val_macro_auroc": best_auc, "device": str(device)})
        if val_auc > best_auc + 1e-8:
            best_auc, best_epoch = val_auc, epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            stale = 0
        else: stale += 1
        if epoch >= MIN_EPOCHS and stale >= PATIENCE: break
    if best_state is None: raise RuntimeError("linear probe produced no checkpoint")
    model.load_state_dict(best_state)
    test_prob = predict(model, x_test, device)
    test_auc, test_auprc, per_class = macro_metrics(y_test, test_prob)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "mean": mean, "std": std}, output / "best.pt")
    np.savez_compressed(output / "predictions.npz", y_true=y_test, y_prob=test_prob,
                        record_indices=test_record_indices)
    atomic_json(output / "history.json", history)
    subset_hash = hashlib.sha256(train_record_indices[selection].tobytes()).hexdigest()
    result = {"state": "complete", "protocol": "pytorch-linear-v1", "model": model_name,
        "task": task, "fraction": fraction, "seed": seed, "train_records": count,
        "training_subset_sha256": subset_hash, "best_epoch": best_epoch,
        "best_val_macro_auroc": best_auc, "test_macro_auroc": test_auc,
        "test_macro_auprc": test_auprc, "per_class": per_class,
        "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE, "selection_metric": "validation_macro_auroc",
        "encoder_frozen": True, "device": str(device)}
    atomic_json(done, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args(); root = args.root.resolve(); device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("GPU required")
    campaign = root / "results" / CAMPAIGN; status = campaign / f"seed-{args.seed}-status.json"
    completed = 0
    for model_name in MODELS:
        for task in TASKS:
            for fraction in FRACTIONS:
                atomic_json(status, {"state": "running", "seed": args.seed, "model": model_name,
                    "task": task, "fraction": fraction, "completed_units": completed,
                    "total_units": 36, "device": str(device)})
                train_unit(root, model_name, task, fraction, args.seed, device); completed += 1
    value = {"state": "complete", "seed": args.seed, "completed_units": completed,
             "total_units": 36, "device": str(device)}
    atomic_json(campaign / f"seed-{args.seed}-complete.json", value); atomic_json(status, value)


if __name__ == "__main__": main()
