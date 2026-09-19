#!/usr/bin/env python3
"""Train only LRA on top of a frozen validation-selected HILA-K checkpoint."""
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

from model import HILAKLRA
from protocol import (BASELINE_CAMPAIGN, BATCH_SIZE, CAMPAIGN, FRACTION,
                      GATE_INITIAL_VALUE, HILAK_CAMPAIGN, LEARNING_RATE,
                      MAX_EPOCHS, MIN_EPOCHS, PATIENCE, SEED, TASKS,
                      WEIGHT_DECAY)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = path.with_suffix(path.suffix + ".incoming")
    incoming.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(incoming, path)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
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
    def __init__(self, global_features, lead_features, local_features, local_masks,
                 labels, mean: np.ndarray, std: np.ndarray) -> None:
        self.global_features, self.lead_features = global_features, lead_features
        self.local_features, self.local_masks = local_features, local_masks
        self.labels, self.mean, self.std = labels, mean, std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        global_features = np.asarray(self.global_features[index], dtype=np.float32)
        lead_features = np.asarray(self.lead_features[index], dtype=np.float32)
        local_features = np.asarray(self.local_features[index], dtype=np.float32)
        return ((global_features - self.mean) / self.std,
                (lead_features - self.mean[None, :]) / self.std[None, :],
                (local_features - self.mean[None, :]) / self.std[None, :],
                np.asarray(self.local_masks[index], dtype=np.bool_),
                np.asarray(self.labels[index], dtype=np.float32))


def load_split(root: Path, task: str, split: str):
    baseline = root / "results" / BASELINE_CAMPAIGN / "shared/embeddings/TolerantECG" / task / split
    keep = root / "results" / HILAK_CAMPAIGN / "shared/masked_features" / task / split
    local = root / "results" / CAMPAIGN / "shared/local_features" / task / split
    keep_manifest = json.loads((keep / "complete.json").read_text(encoding="utf-8"))
    local_manifest = json.loads((local / "complete.json").read_text(encoding="utf-8"))
    for key in ("source_indices_sha256", "source_labels_sha256"):
        if keep_manifest[key] != local_manifest[key]:
            raise RuntimeError(f"HILA/local {key} mismatch for {task}/{split}")
    arrays = (np.load(baseline / "features.npy", mmap_mode="r"),
              np.load(keep / "features.npy", mmap_mode="r"),
              np.load(local / "features.npy", mmap_mode="r"),
              np.load(local / "masks.npy", mmap_mode="r"),
              (np.load(baseline / "labels.npy", mmap_mode="r") > 0).astype(np.float32))
    if len({len(value) for value in arrays}) != 1:
        raise RuntimeError(f"feature length mismatch for {task}/{split}")
    return (*arrays, keep_manifest, local_manifest)


def checkpoints(root: Path, task: str):
    baseline_folder = root / "results" / BASELINE_CAMPAIGN / "TolerantECG" / f"seed-{SEED}" / task / f"{FRACTION:g}"
    baseline_checkpoint = torch.load(baseline_folder / "best.pt", map_location="cpu", weights_only=False)
    mean = np.asarray(baseline_checkpoint["mean"], dtype=np.float32)
    std = np.asarray(baseline_checkpoint["std"], dtype=np.float32)
    hilak_folder = root / "results" / HILAK_CAMPAIGN / task
    hilak_checkpoint = torch.load(hilak_folder / "best.pt", map_location="cpu", weights_only=True)
    hilak_result = json.loads((hilak_folder / "complete.json").read_text(encoding="utf-8"))
    return baseline_checkpoint["state_dict"], hilak_checkpoint["state_dict"], mean, std, hilak_result


@torch.no_grad()
def predict(model: nn.Module, dataset: Dataset, device: torch.device) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
    outputs = []
    model.eval()
    for global_f, lead_f, local_f, local_mask, _ in loader:
        logits = model(global_f.to(device, non_blocking=True), lead_f.to(device, non_blocking=True),
                       local_f.to(device, non_blocking=True), local_mask.to(device, non_blocking=True))
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
    train_values, val_values = load_split(root, task, "train"), load_split(root, task, "val")
    train_global, train_lead, train_local, train_mask, train_labels, train_k, train_l = train_values
    val_global, val_lead, val_local, val_mask, val_labels, val_k, val_l = val_values
    baseline_state, hilak_state, mean, std, hilak_result = checkpoints(root, task)
    seed_everything(SEED)
    train_dataset = FeatureDataset(train_global, train_lead, train_local, train_mask, train_labels, mean, std)
    val_dataset = FeatureDataset(val_global, val_lead, val_local, val_mask, val_labels, mean, std)
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED), num_workers=0,
                        pin_memory=True, drop_last=False)
    model = HILAKLRA(train_labels.shape[1], baseline_state, hilak_state).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    initial_prob = predict(model, val_dataset, device)
    initial_auc, initial_auprc = macro_metrics(np.asarray(val_labels), initial_prob)
    if abs(initial_auc - hilak_result["best_val_macro_auroc"]) > 1e-7:
        raise RuntimeError(f"frozen HILA-K reproduction mismatch: {initial_auc} vs {hilak_result['best_val_macro_auroc']}")
    history = [{"epoch": 0, "learning_rate": 0.0, "train_loss": None,
                "val_macro_auroc": initial_auc, "val_macro_auprc": initial_auprc}]
    best_auc, best_epoch, stale = initial_auc, 0, 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); loss_sum = 0.0
        for global_f, lead_f, local_f, local_mask, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(global_f.to(device, non_blocking=True), lead_f.to(device, non_blocking=True),
                           local_f.to(device, non_blocking=True), local_mask.to(device, non_blocking=True))
            loss = criterion(logits, labels.to(device, non_blocking=True))
            loss.backward(); optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
        val_prob = predict(model, val_dataset, device)
        val_auc, val_auprc = macro_metrics(np.asarray(val_labels), val_prob)
        history.append({"epoch": epoch, "learning_rate": LEARNING_RATE,
                        "train_loss": loss_sum / len(train_dataset),
                        "val_macro_auroc": val_auc, "val_macro_auprc": val_auprc})
        if val_auc > best_auc + 1e-8:
            best_auc, best_epoch, stale = val_auc, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        atomic_json(output / "status.json", {"state": "training", "task": task,
            "epoch": epoch, "best_epoch": best_epoch,
            "best_val_macro_auroc": best_auc, "stale_epochs": stale})
        if epoch >= MIN_EPOCHS and stale >= PATIENCE:
            break
    torch.save({"state_dict": best_state, "epoch": best_epoch}, output / "best.pt")
    atomic_json(output / "history.json", history)
    result = {"state": "complete", "protocol": "tolerant-hilak-lra-ecgfix-validation-screen-v1",
        "task": task, "seed": SEED, "fraction": FRACTION,
        "train_records": len(train_dataset), "val_records": len(val_dataset),
        "best_epoch": best_epoch, "epochs_ran": len(history) - 1,
        "best_val_macro_auroc": best_auc,
        "best_val_macro_auprc": history[best_epoch]["val_macro_auprc"],
        "hilak_val_macro_auroc": initial_auc, "val_auroc_delta_vs_hilak": best_auc - initial_auc,
        "test_evaluations": 0, "test_data_loaded": False,
        "encoder_frozen": True, "baseline_head_frozen": True, "hilak_frozen": True,
        "local_head_zero_initialized": True, "local_gate_initial_value": GATE_INITIAL_VALUE,
        "local_architecture": "768-512-256-128; masked heartbeat mean; residual logits",
        "local_feature_standardization": "same baseline train mean/std",
        "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE, "scheduler": "none (matching HILA-K)",
        "selection_metric": "validation_macro_auroc including frozen HILA-K at epoch 0",
        "train_keep_manifest": train_k, "val_keep_manifest": val_k,
        "train_local_manifest": train_l, "val_local_manifest": val_l}
    atomic_json(complete, result); atomic_json(output / "status.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args(); device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    print(json.dumps(train(args.root.resolve(), args.task, device), indent=2))


if __name__ == "__main__":
    main()
