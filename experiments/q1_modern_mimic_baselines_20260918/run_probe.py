#!/usr/bin/env python3
"""Train validation-selected PyTorch linear probes on frozen modern ECG embeddings."""
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
from torch.utils.data import DataLoader, TensorDataset

from prepare_embeddings import CAMPAIGN, DATASETS, MODELS, atomic_json


FRACTIONS = (0.01, 0.1, 1.0)
SEEDS = (42, 46, 55)
MAX_EPOCHS = 100
MIN_EPOCHS = 10
PATIENCE = 12
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split(root: Path, model_name: str, dataset: str, split: str):
    folder = root / "results" / CAMPAIGN / "shared/embeddings" / model_name / dataset / split
    manifest = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    x = np.load(folder / "features.npy", mmap_mode="r")
    y = (np.load(folder / "labels.npy", mmap_mode="r") > 0).astype(np.float32)
    return x, y, manifest


def macro_metrics(y_true: np.ndarray, y_prob: np.ndarray):
    rows, aucs, auprcs = [], [], []
    for index in range(y_true.shape[1]):
        truth = y_true[:, index]
        auc = None if np.unique(truth).size < 2 else float(roc_auc_score(truth, y_prob[:, index]))
        if auc is not None:
            aucs.append(auc)
        auprc = float(average_precision_score(truth, y_prob[:, index]))
        auprcs.append(auprc)
        rows.append({"class_index": index, "auroc": auc, "auprc": auprc,
                     "positives": int(truth.sum()), "total": int(truth.size)})
    return float(np.mean(aucs)), float(np.mean(auprcs)), rows


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, mean: np.ndarray, std: np.ndarray,
            device: torch.device, batch_size: int = 4096) -> np.ndarray:
    outputs = []
    model.eval()
    for start in range(0, len(x), batch_size):
        batch = np.asarray(x[start:start + batch_size], dtype=np.float32)
        batch = (batch - mean) / std
        logits = model(torch.from_numpy(batch).to(device))
        outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def train_unit(root: Path, model_name: str, dataset: str, fraction: float, seed: int,
               device: torch.device) -> dict:
    output = root / "results" / CAMPAIGN / model_name / f"seed-{seed}" / dataset / f"{fraction:g}"
    done = output / "complete.json"
    if done.is_file():
        value = json.loads(done.read_text(encoding="utf-8"))
        if value.get("state") == "complete" and value.get("protocol") == "frozen-pytorch-linear-v1":
            return value
    x_train, y_train, train_meta = load_split(root, model_name, dataset, "train")
    x_val, y_val, val_meta = load_split(root, model_name, dataset, "val")
    x_test, y_test, test_meta = load_split(root, model_name, dataset, "test")
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    count = max(1, int(len(x_train) * fraction))
    indices = torch.randperm(len(x_train), generator=generator)[:count].numpy()
    selected = np.asarray(x_train[indices], dtype=np.float32)
    selected_y = np.asarray(y_train[indices], dtype=np.float32)
    mean = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = selected.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    selected = (selected - mean) / std
    loader = DataLoader(TensorDataset(torch.from_numpy(selected), torch.from_numpy(selected_y)),
                        batch_size=BATCH_SIZE, shuffle=True, generator=generator,
                        num_workers=0, pin_memory=True)
    model = nn.Linear(selected.shape[1], selected_y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    best_auc, best_epoch, best_state, stale = -float("inf"), 0, None, 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); loss_sum = 0.0; examples = 0
        for features, labels in loader:
            features = features.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels); loss.backward(); optimizer.step()
            loss_sum += float(loss.detach()) * len(features); examples += len(features)
        val_prob = predict(model, x_val, mean, std, device)
        val_auc, val_auprc, _ = macro_metrics(y_val, val_prob)
        history.append({"epoch": epoch, "train_loss": loss_sum / examples,
                        "val_macro_auroc": val_auc, "val_macro_auprc": val_auprc})
        atomic_json(output / "status.json", {
            "state": "training", "model": model_name, "dataset": dataset,
            "fraction": fraction, "seed": seed, "epoch": epoch,
            "best_epoch": best_epoch, "best_val_macro_auroc": best_auc,
        })
        if val_auc > best_auc + 1e-8:
            best_auc, best_epoch = val_auc, epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= MIN_EPOCHS and stale >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("linear probe did not produce a checkpoint")
    model.load_state_dict(best_state)
    test_prob = predict(model, x_test, mean, std, device)
    test_auc, test_auprc, per_class = macro_metrics(y_test, test_prob)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "mean": mean, "std": std}, output / "best.pt")
    np.savez_compressed(output / "predictions.npz", y_true=y_test, y_prob=test_prob)
    atomic_json(output / "history.json", history)
    result = {
        "state": "complete", "protocol": "frozen-pytorch-linear-v1", "model": model_name,
        "dataset": dataset, "fraction": fraction, "seed": seed, "train_records": int(count),
        "train_subset_indices": indices.tolist(), "best_epoch": best_epoch,
        "best_val_macro_auroc": best_auc, "test_macro_auroc": test_auc,
        "test_macro_auprc": test_auprc, "per_class": per_class,
        "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE, "selection_metric": "validation_macro_auroc",
        "checkpoint_sha256": train_meta["checkpoint_sha256"],
        "train_source_indices_sha256": train_meta["source_indices_sha256"],
        "val_source_indices_sha256": val_meta["source_indices_sha256"],
        "test_source_indices_sha256": test_meta["source_indices_sha256"],
        "train_labels_sha256": train_meta["labels_sha256"],
        "val_labels_sha256": val_meta["labels_sha256"],
        "test_labels_sha256": test_meta["labels_sha256"],
    }
    atomic_json(done, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    root = args.root.resolve(); device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required")
    completed = 0
    campaign = root / "results" / CAMPAIGN
    status = campaign / f"{args.model}-status.json"
    for seed in SEEDS:
        for dataset in DATASETS:
            for fraction in FRACTIONS:
                atomic_json(status, {"state": "probing", "model": args.model, "seed": seed,
                                     "dataset": dataset, "fraction": fraction,
                                     "completed_units": completed, "total_units": 54})
                train_unit(root, args.model, dataset, fraction, seed, device)
                completed += 1
    value = {"state": "complete", "model": args.model, "completed_units": completed,
             "total_units": 54}
    atomic_json(campaign / f"{args.model}-complete.json", value)
    atomic_json(status, value)


if __name__ == "__main__":
    main()
