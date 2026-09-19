#!/usr/bin/env python3
"""Train locked HILA-K then LRA for one task/seed; validation only."""
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

from model import HILAK, HILAKLRA
from protocol import (BASELINE_CAMPAIGN, BATCH_SIZE, CAMPAIGN, FRACTION,
                      GATE_INITIAL_VALUE, LEAD_FEATURE_CAMPAIGN, LEARNING_RATE,
                      LOCAL_FEATURE_CAMPAIGN, MAX_EPOCHS, MIN_EPOCHS, PATIENCE,
                      SEEDS, TASKS, WEIGHT_DECAY)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = path.with_suffix(path.suffix + ".incoming")
    incoming.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(incoming, path)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def macro_metrics(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    aucs, aps = [], []
    for index in range(labels.shape[1]):
        truth = labels[:, index]
        if np.unique(truth).size == 2:
            aucs.append(roc_auc_score(truth, probabilities[:, index]))
        aps.append(average_precision_score(truth, probabilities[:, index]))
    return float(np.mean(aucs)), float(np.mean(aps))


class Features(Dataset):
    def __init__(self, global_f, lead_f, local_f, masks, labels,
                 mean: np.ndarray, std: np.ndarray) -> None:
        self.global_f, self.lead_f, self.local_f = global_f, lead_f, local_f
        self.masks, self.labels, self.mean, self.std = masks, labels, mean, std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        global_f = np.asarray(self.global_f[index], dtype=np.float32)
        lead_f = np.asarray(self.lead_f[index], dtype=np.float32)
        local_f = np.asarray(self.local_f[index], dtype=np.float32)
        return ((global_f - self.mean) / self.std,
                (lead_f - self.mean[None, :]) / self.std[None, :],
                (local_f - self.mean[None, :]) / self.std[None, :],
                np.asarray(self.masks[index], dtype=np.bool_),
                np.asarray(self.labels[index], dtype=np.float32))


def load_split(root: Path, task: str, split: str):
    base = root / "results" / BASELINE_CAMPAIGN / "shared/embeddings/TolerantECG" / task / split
    lead = root / "results" / LEAD_FEATURE_CAMPAIGN / "shared/masked_features" / task / split
    local = root / "results" / LOCAL_FEATURE_CAMPAIGN / "shared/local_features" / task / split
    base_manifest = json.loads((base / "complete.json").read_text(encoding="utf-8"))
    lead_manifest = json.loads((lead / "complete.json").read_text(encoding="utf-8"))
    local_manifest = json.loads((local / "complete.json").read_text(encoding="utf-8"))
    base_label_hash = base_manifest["labels_sha256"]
    if len({base_manifest["source_indices_sha256"], lead_manifest["source_indices_sha256"],
            local_manifest["source_indices_sha256"]}) != 1:
        raise RuntimeError(f"source index mismatch for {task}/{split}")
    if len({base_label_hash, lead_manifest["source_labels_sha256"],
            local_manifest["source_labels_sha256"]}) != 1:
        raise RuntimeError(f"label mismatch for {task}/{split}")
    arrays = (np.load(base / "features.npy", mmap_mode="r"),
              np.load(lead / "features.npy", mmap_mode="r"),
              np.load(local / "features.npy", mmap_mode="r"),
              np.load(local / "masks.npy", mmap_mode="r"),
              (np.load(base / "labels.npy", mmap_mode="r") > 0).astype(np.float32))
    if len({len(value) for value in arrays}) != 1:
        raise RuntimeError(f"feature length mismatch for {task}/{split}")
    return arrays


def baseline(root: Path, task: str, seed: int):
    folder = root / "results" / BASELINE_CAMPAIGN / "TolerantECG" / f"seed-{seed}" / task / f"{FRACTION:g}"
    complete = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
    return (checkpoint["state_dict"], np.asarray(checkpoint["mean"], dtype=np.float32),
            np.asarray(checkpoint["std"], dtype=np.float32), complete)


@torch.no_grad()
def predict(model: nn.Module, dataset: Dataset, device: torch.device, stage: str) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
    outputs = []
    model.eval()
    for global_f, lead_f, local_f, local_mask, _ in loader:
        global_f, lead_f = global_f.to(device, non_blocking=True), lead_f.to(device, non_blocking=True)
        if stage == "hila-k":
            logits = model(global_f, lead_f)
        else:
            logits = model(global_f, lead_f, local_f.to(device, non_blocking=True),
                           local_mask.to(device, non_blocking=True))
        outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs)


def datasets(root: Path, task: str, mean: np.ndarray, std: np.ndarray):
    train_values, val_values = load_split(root, task, "train"), load_split(root, task, "val")
    return (Features(*train_values, mean, std), Features(*val_values, mean, std))


def train_hilak(root: Path, task: str, seed: int, device: torch.device) -> dict:
    output = root / "results" / CAMPAIGN / f"seed-{seed}" / task / "hila-k"
    complete_path = output / "complete.json"
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("state") == "complete": return complete
    base_state, mean, std, base_result = baseline(root, task, seed)
    seed_everything(seed)
    train_set, val_set = datasets(root, task, mean, std)
    loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0,
                        pin_memory=True, drop_last=False)
    model = HILAK(train_set.labels.shape[1], base_state).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(); history = []
    best_auc, best_epoch, best_state, stale = -float("inf"), 0, None, 0
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); loss_sum = 0.0
        for global_f, lead_f, _, _, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(global_f.to(device), lead_f.to(device)), labels.to(device))
            loss.backward(); optimizer.step(); loss_sum += float(loss.detach()) * len(labels)
        probabilities = predict(model, val_set, device, "hila-k")
        auc, ap = macro_metrics(np.asarray(val_set.labels), probabilities)
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train_set),
                        "val_macro_auroc": auc, "val_macro_auprc": ap})
        if auc > best_auc + 1e-8:
            best_auc, best_epoch, stale = auc, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else: stale += 1
        atomic_json(output / "status.json", {"state": "training", "stage": "hila-k",
                    "seed": seed, "task": task, "epoch": epoch, "best_epoch": best_epoch,
                    "best_val_macro_auroc": best_auc, "stale_epochs": stale})
        if epoch >= MIN_EPOCHS and stale >= PATIENCE: break
    torch.save({"state_dict": best_state, "epoch": best_epoch}, output / "best.pt")
    atomic_json(output / "history.json", history)
    result = {"state": "complete", "stage": "hila-k", "task": task, "seed": seed,
              "fraction": FRACTION, "best_epoch": best_epoch, "epochs_ran": len(history),
              "best_val_macro_auroc": best_auc,
              "best_val_macro_auprc": history[best_epoch - 1]["val_macro_auprc"],
              "baseline_val_macro_auroc": base_result["best_val_macro_auroc"],
              "test_data_loaded": False, "test_evaluations": 0,
              "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
              "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
              "max_epochs": MAX_EPOCHS, "early_stopping_patience": PATIENCE,
              "scheduler": "none", "selection_metric": "validation_macro_auroc"}
    atomic_json(complete_path, result); atomic_json(output / "status.json", result)
    return result


def train_lra(root: Path, task: str, seed: int, device: torch.device) -> dict:
    output = root / "results" / CAMPAIGN / f"seed-{seed}" / task / "hila-k-lra"
    complete_path = output / "complete.json"
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("state") == "complete": return complete
    base_state, mean, std, _ = baseline(root, task, seed)
    hila_folder = root / "results" / CAMPAIGN / f"seed-{seed}" / task / "hila-k"
    hila_checkpoint = torch.load(hila_folder / "best.pt", map_location="cpu", weights_only=True)
    hila_result = json.loads((hila_folder / "complete.json").read_text(encoding="utf-8"))
    seed_everything(seed)
    train_set, val_set = datasets(root, task, mean, std)
    loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0,
                        pin_memory=True, drop_last=False)
    model = HILAKLRA(train_set.labels.shape[1], base_state, hila_checkpoint["state_dict"]).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    initial_prob = predict(model, val_set, device, "hila-k-lra")
    initial_auc, initial_ap = macro_metrics(np.asarray(val_set.labels), initial_prob)
    if abs(initial_auc - hila_result["best_val_macro_auroc"]) > 1e-7:
        raise RuntimeError(f"frozen HILA reproduction mismatch: {initial_auc}")
    history = [{"epoch": 0, "train_loss": None, "val_macro_auroc": initial_auc,
                "val_macro_auprc": initial_ap}]
    best_auc, best_epoch, stale = initial_auc, 0, 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); loss_sum = 0.0
        for global_f, lead_f, local_f, local_mask, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(global_f.to(device), lead_f.to(device), local_f.to(device), local_mask.to(device))
            loss = criterion(logits, labels.to(device)); loss.backward(); optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
        probabilities = predict(model, val_set, device, "hila-k-lra")
        auc, ap = macro_metrics(np.asarray(val_set.labels), probabilities)
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train_set),
                        "val_macro_auroc": auc, "val_macro_auprc": ap})
        if auc > best_auc + 1e-8:
            best_auc, best_epoch, stale = auc, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else: stale += 1
        atomic_json(output / "status.json", {"state": "training", "stage": "hila-k-lra",
                    "seed": seed, "task": task, "epoch": epoch, "best_epoch": best_epoch,
                    "best_val_macro_auroc": best_auc, "stale_epochs": stale})
        if epoch >= MIN_EPOCHS and stale >= PATIENCE: break
    torch.save({"state_dict": best_state, "epoch": best_epoch}, output / "best.pt")
    atomic_json(output / "history.json", history)
    result = {"state": "complete", "stage": "hila-k-lra", "task": task, "seed": seed,
              "fraction": FRACTION, "best_epoch": best_epoch, "epochs_ran": len(history) - 1,
              "best_val_macro_auroc": best_auc,
              "best_val_macro_auprc": history[best_epoch]["val_macro_auprc"],
              "hilak_val_macro_auroc": initial_auc,
              "test_data_loaded": False, "test_evaluations": 0,
              "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
              "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
              "max_epochs": MAX_EPOCHS, "early_stopping_patience": PATIENCE,
              "scheduler": "none", "selection_metric": "validation_macro_auroc including epoch 0"}
    atomic_json(complete_path, result); atomic_json(output / "status.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    if args.seed == 42: raise RuntimeError("seed 42 is already complete and must be reused")
    root = args.root.resolve()
    print(json.dumps(train_hilak(root, args.task, args.seed, device), indent=2))
    print(json.dumps(train_lra(root, args.task, args.seed, device), indent=2))


if __name__ == "__main__":
    main()
