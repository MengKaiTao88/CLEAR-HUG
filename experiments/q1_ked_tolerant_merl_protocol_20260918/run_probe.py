#!/usr/bin/env python3
"""Train MERL-split frozen linear probes for KED and TolerantECG."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from protocol import (BATCH_SIZE, CAMPAIGN, LEARNING_RATE, MAX_EPOCHS, MIN_EPOCHS,
                      MODELS, PATIENCE, RATIOS, SEEDS, TASKS, WARMUP_EPOCHS,
                      WEIGHT_DECAY)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = path.with_suffix(path.suffix + ".incoming")
    incoming.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(incoming, path)


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def macro_metrics(y_true: np.ndarray, y_prob: np.ndarray):
    rows, aucs, auprcs = [], [], []
    for index in range(y_true.shape[1]):
        truth = y_true[:, index]
        auc = None if np.unique(truth).size < 2 else float(roc_auc_score(truth, y_prob[:, index]))
        auprc = float(average_precision_score(truth, y_prob[:, index]))
        if auc is not None: aucs.append(auc)
        auprcs.append(auprc)
        rows.append({"class_index": index, "auroc": auc, "auprc": auprc,
                     "positives": int(truth.sum()), "total": int(truth.size)})
    if not aucs:
        raise RuntimeError("no validation classes contain both labels")
    return float(np.mean(aucs)), float(np.mean(auprcs)), rows


@torch.no_grad()
def predict(model: nn.Module, values: np.ndarray, device: torch.device, batch_size: int = 4096):
    outputs = []; model.eval()
    for start in range(0, len(values), batch_size):
        batch = torch.from_numpy(np.asarray(values[start:start + batch_size], dtype=np.float32)).to(device)
        outputs.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def learning_rate(epoch: int) -> float:
    if epoch <= WARMUP_EPOCHS:
        return LEARNING_RATE * epoch / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, MAX_EPOCHS - WARMUP_EPOCHS)
    return LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))


def subset_indices(records: int, ratio: int, seed: int) -> np.ndarray:
    indices = np.arange(records, dtype=np.int64)
    if ratio == 100:
        return indices
    selected, _ = train_test_split(indices, train_size=ratio / 100, random_state=seed)
    return np.asarray(selected, dtype=np.int64)


def load_split(root: Path, model_name: str, task: str, split: str):
    folder = root / "results" / CAMPAIGN / "shared/merl_embeddings" / model_name / task / split
    manifest = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    if manifest.get("state") != "complete": raise RuntimeError(f"incomplete {folder}")
    return (np.load(folder / "features.npy", mmap_mode="r"),
            np.load(folder / "labels.npy", mmap_mode="r"),
            np.load(folder / "record_ids.npy", mmap_mode="r"), manifest)


def train_unit(root: Path, model_name: str, task: str, ratio: int, seed: int,
               device: torch.device, max_epochs: int = MAX_EPOCHS,
               output_root: Path | None = None) -> dict:
    campaign_root = output_root or (root / "results" / CAMPAIGN)
    output = campaign_root / model_name / task / f"seed{seed}" / f"{ratio}pct"
    complete = output / "complete.json"
    if complete.is_file():
        value = json.loads(complete.read_text(encoding="utf-8"))
        if value.get("state") == "complete" and value.get("protocol") == "merl-split-frozen-linear-v1":
            return value
    x_train, y_train, train_ids, train_meta = load_split(root, model_name, task, "train")
    x_val, y_val, val_ids, val_meta = load_split(root, model_name, task, "val")
    x_test, y_test, test_ids, test_meta = load_split(root, model_name, task, "test")
    indices = subset_indices(len(x_train), ratio, seed)
    selected_x = np.asarray(x_train[indices], dtype=np.float32)
    selected_y = np.asarray(y_train[indices], dtype=np.float32)
    selected_ids = np.asarray(train_ids[indices])
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(selected_x), torch.from_numpy(selected_y)),
                        batch_size=BATCH_SIZE, shuffle=True, generator=generator,
                        num_workers=0, pin_memory=True, drop_last=False)
    model = nn.Linear(selected_x.shape[1], selected_y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    best_auc, best_epoch, best_state, stale = -float("inf"), 0, None, 0
    history = []
    output.mkdir(parents=True, exist_ok=True)
    subset_manifest = {
        "task": task, "ratio": ratio, "seed": seed, "source_records": len(x_train),
        "selected_records": len(indices), "indices_sha256": array_sha256(indices),
        "record_ids_sha256": array_sha256(selected_ids),
        "selected_labels_sha256": array_sha256(selected_y),
        "selection_implementation": "sklearn.model_selection.train_test_split(train_size=ratio/100, random_state=seed)",
    }
    atomic_json(output / "subset-manifest.json", subset_manifest)
    for epoch in range(1, max_epochs + 1):
        lr = learning_rate(epoch)
        for group in optimizer.param_groups: group["lr"] = lr
        model.train(); loss_sum = 0.0; examples = 0
        for features, labels in loader:
            features = features.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels); loss.backward(); optimizer.step()
            loss_sum += float(loss.detach()) * len(features); examples += len(features)
        val_prob = predict(model, x_val, device)
        val_auc, val_auprc, _ = macro_metrics(np.asarray(y_val), val_prob)
        row = {"epoch": epoch, "learning_rate": lr, "train_loss": loss_sum / examples,
               "val_macro_auroc": val_auc, "val_macro_auprc": val_auprc}
        history.append(row)
        if val_auc > best_auc + 1e-8:
            best_auc, best_epoch = val_auc, epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        atomic_json(output / "status.json", {"state": "training", "model": model_name,
            "task": task, "ratio": ratio, "seed": seed, "epoch": epoch,
            "best_epoch": best_epoch, "best_val_macro_auroc": best_auc, "stale_epochs": stale})
        if max_epochs == MAX_EPOCHS and epoch >= MIN_EPOCHS and stale >= PATIENCE:
            break
    if best_state is None: raise RuntimeError("probe produced no checkpoint")
    model.load_state_dict(best_state)
    test_prob = predict(model, x_test, device)
    test_auc, test_auprc, per_class = macro_metrics(np.asarray(y_test), test_prob)
    torch.save({"state_dict": best_state, "epoch": best_epoch}, output / "best.pt")
    np.savez_compressed(output / "test-predictions.npz", y_true=np.asarray(y_test),
                        y_prob=test_prob, record_ids=np.asarray(test_ids))
    atomic_json(output / "history.json", history)
    result = {
        "state": "complete", "protocol": "merl-split-frozen-linear-v1",
        "model": model_name, "task": task, "ratio": ratio, "seed": seed,
        "train_records": len(indices), "best_epoch": best_epoch,
        "epochs_ran": len(history), "best_val_macro_auroc": best_auc,
        "test_macro_auroc": test_auc, "test_macro_auprc": test_auprc,
        "per_class": per_class, "encoder_frozen": True,
        "head": f"Linear({selected_x.shape[1]},{selected_y.shape[1]})",
        "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
        "max_epochs": max_epochs, "min_epochs": MIN_EPOCHS,
        "early_stopping_patience": PATIENCE, "warmup_epochs": WARMUP_EPOCHS,
        "scheduler": "5-epoch linear warmup then cosine annealing",
        "selection_metric": "validation_macro_auroc", "test_evaluations": 1,
        "feature_standardization": False,
        "subset_indices_sha256": subset_manifest["indices_sha256"],
        "subset_record_ids_sha256": subset_manifest["record_ids_sha256"],
        "train_split_manifest": train_meta, "val_split_manifest": val_meta,
        "test_split_manifest": test_meta,
    }
    atomic_json(complete, result); atomic_json(output / "status.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = parser.parse_args(); root = args.root.resolve(); device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    campaign = root / "results" / CAMPAIGN; completed = 0
    total = len(args.models) * len(TASKS) * len(RATIOS)
    status = campaign / f"seed-{args.seed}-status.json"
    for model_name in args.models:
        for task in TASKS:
            for ratio in RATIOS:
                atomic_json(status, {"state": "running", "seed": args.seed, "model": model_name,
                    "task": task, "ratio": ratio, "completed_units": completed,
                    "total_units": total, "device": str(device)})
                train_unit(root, model_name, task, ratio, args.seed, device); completed += 1
    value = {"state": "complete", "seed": args.seed, "completed_units": completed,
             "total_units": total, "device": str(device)}
    atomic_json(campaign / f"seed-{args.seed}-complete.json", value); atomic_json(status, value)


if __name__ == "__main__":
    main()
