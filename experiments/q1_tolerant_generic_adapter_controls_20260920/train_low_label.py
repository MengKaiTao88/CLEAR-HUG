#!/usr/bin/env python3
"""Train locked global-only controls on exact 1%/10% baseline subsets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from model import GenericResidualAdapter, ParameterMatchedMLP, trainable_parameters
from protocol import (BASELINE_CAMPAIGN, BATCH_SIZE, LEARNING_RATE,
                      LOW_LABEL_CAMPAIGN, MAX_EPOCHS, METHODS, MIN_EPOCHS,
                      PATIENCE, SEEDS, TASKS, WEIGHT_DECAY,
                      hilar_parameter_budget)

FRACTIONS = (0.01, 0.1)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = path.with_suffix(path.suffix + ".incoming")
    incoming.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(incoming, path)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def indices_sha256(indices: list[int]) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def macro_metrics(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    aucs, aps = [], []
    for index in range(labels.shape[1]):
        truth = labels[:, index]
        if np.unique(truth).size == 2:
            aucs.append(roc_auc_score(truth, probabilities[:, index]))
        aps.append(average_precision_score(truth, probabilities[:, index]))
    return float(np.mean(aucs)), float(np.mean(aps))


class GlobalFeatures(Dataset):
    def __init__(self, features, labels, mean: np.ndarray, std: np.ndarray) -> None:
        self.features, self.labels = features, labels
        self.mean, self.std = mean, std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        features = np.asarray(self.features[index], dtype=np.float32)
        labels = np.asarray(self.labels[index], dtype=np.float32)
        return (features - self.mean) / self.std, labels


def load_split(root: Path, task: str, split: str):
    folder = (root / "results" / BASELINE_CAMPAIGN / "shared/embeddings/TolerantECG"
              / task / split)
    return (np.load(folder / "features.npy", mmap_mode="r"),
            (np.load(folder / "labels.npy", mmap_mode="r") > 0).astype(np.float32))


def load_baseline(root: Path, task: str, seed: int, fraction: float):
    folder = (root / "results" / BASELINE_CAMPAIGN / "TolerantECG"
              / f"seed-{seed}" / task / f"{fraction:g}")
    result = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
    if abs(float(result["fraction"]) - fraction) > 1e-12:
        raise RuntimeError(f"baseline fraction mismatch: {folder}")
    return checkpoint, result


@torch.no_grad()
def predict(model: nn.Module, dataset: Dataset, device: torch.device) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0,
                        pin_memory=True)
    outputs = []; model.eval()
    for features, _ in loader:
        outputs.append(torch.sigmoid(model(features.to(device, non_blocking=True))).cpu().numpy())
    return np.concatenate(outputs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--fraction", choices=FRACTIONS, type=float, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); root = args.root.resolve(); device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    output = (root / "results" / LOW_LABEL_CAMPAIGN / f"seed-{args.seed}"
              / args.task / f"{args.fraction:g}" / args.method)
    complete_path = output / "complete.json"
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("state") == "complete":
            print(json.dumps(complete, indent=2)); return

    baseline_checkpoint, baseline_result = load_baseline(root, args.task, args.seed, args.fraction)
    subset_indices = [int(value) for value in baseline_result["train_subset_indices"]]
    train_values, val_values = load_split(root, args.task, "train"), load_split(root, args.task, "val")
    mean = np.asarray(baseline_checkpoint["mean"], dtype=np.float32)
    std = np.asarray(baseline_checkpoint["std"], dtype=np.float32)
    full_train = GlobalFeatures(*train_values, mean, std)
    if not subset_indices or min(subset_indices) < 0 or max(subset_indices) >= len(full_train):
        raise RuntimeError("invalid baseline train subset indices")
    if len(set(subset_indices)) != len(subset_indices):
        raise RuntimeError("duplicate baseline train subset indices")
    train_set = Subset(full_train, subset_indices)
    val_set = GlobalFeatures(*val_values, mean, std)
    classes = val_set.labels.shape[1]
    seed_everything(args.seed)
    if args.method == "parameter-matched-mlp":
        model = ParameterMatchedMLP(classes)
    else:
        model = GenericResidualAdapter(classes, baseline_checkpoint["state_dict"])
    actual_parameters, target_parameters = trainable_parameters(model), hilar_parameter_budget(classes)
    mismatch_percent = 100.0 * (actual_parameters - target_parameters) / target_parameters
    if abs(mismatch_percent) > 1.0:
        raise RuntimeError(f"parameter mismatch exceeds 1%: {mismatch_percent:.6f}%")
    model.to(device)
    loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed), num_workers=0,
                        pin_memory=True, drop_last=False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(); history = []
    best_auc, best_epoch, best_state, stale = -float("inf"), 0, None, 0
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); loss_sum = 0.0
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features.to(device, non_blocking=True)),
                             labels.to(device, non_blocking=True))
            loss.backward(); optimizer.step(); loss_sum += float(loss.detach()) * len(labels)
        probabilities = predict(model, val_set, device)
        auc, ap = macro_metrics(np.asarray(val_set.labels), probabilities)
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train_set),
                        "val_macro_auroc": auc, "val_macro_auprc": ap})
        if auc > best_auc + 1e-8:
            best_auc, best_epoch, stale = auc, epoch, 0
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        else:
            stale += 1
        status = {"state": "training", "method": args.method, "task": args.task,
                  "seed": args.seed, "fraction": args.fraction, "epoch": epoch,
                  "best_epoch": best_epoch, "best_val_macro_auroc": best_auc,
                  "stale_epochs": stale, "trainable_parameters": actual_parameters,
                  "hilar_parameter_budget": target_parameters,
                  "parameter_mismatch_percent": mismatch_percent,
                  "hidden_width": model.width, "train_records": len(train_set),
                  "train_subset_indices_sha256": indices_sha256(subset_indices)}
        atomic_json(output / "status.json", status); print(json.dumps(status), flush=True)
        if epoch >= MIN_EPOCHS and stale >= PATIENCE:
            break
    torch.save({"state_dict": best_state, "epoch": best_epoch, "mean": mean, "std": std,
                "train_subset_indices_sha256": indices_sha256(subset_indices)}, output / "best.pt")
    atomic_json(output / "history.json", history)
    result = {**status, "state": "complete", "epochs_ran": len(history),
              "best_epoch": best_epoch, "best_val_macro_auroc": best_auc,
              "best_val_macro_auprc": history[best_epoch - 1]["val_macro_auprc"],
              "baseline_val_macro_auroc": baseline_result["best_val_macro_auroc"],
              "test_data_loaded": False, "test_evaluations": 0,
              "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
              "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
              "max_epochs": MAX_EPOCHS, "early_stopping_patience": PATIENCE,
              "scheduler": "none", "selection_metric": "validation_macro_auroc",
              "subset_policy": "exact baseline train_subset_indices"}
    atomic_json(complete_path, result); atomic_json(output / "status.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

