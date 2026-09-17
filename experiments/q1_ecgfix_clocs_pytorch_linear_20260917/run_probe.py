#!/usr/bin/env python3
"""GPU PyTorch linear probes over the frozen ECG-FIX CLOCS embeddings."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CAMPAIGN = "q1-ecgfix-clocs-pytorch-linear-six-task-three-fraction-seeds42-46-55-20260917"
SOURCE_CAMPAIGN = "q1-ecgfix-clocs-six-task-three-fraction-seeds42-46-55-20260916"
ECGFIX_COMMIT = "991a31f14c94f72d5658bc172eae5736ba2fa149"
CLOCS_SHA256 = "039975cf563e76dd25a7975abfe1b74ff37308bd1e9fb24aaf6282f4ffdc5805"
DATASETS = ("PTBXL_form", "PTBXL_super", "PTBXL_sub", "PTBXL_rhythm", "CPSC", "CSN")
FRACTIONS = (0.01, 0.1, 1.0)
SEEDS = (42, 46, 55)

MAX_EPOCHS = 100
MIN_EPOCHS = 10
PATIENCE = 12
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = path.with_suffix(path.suffix + ".incoming")
    incoming.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(incoming, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify(root: Path) -> dict:
    checkpoint = root / "model_weights/ecg-fix/best_weights_clocs"
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != CLOCS_SHA256:
        raise RuntimeError(f"CLOCS checkpoint SHA256 mismatch: {checkpoint_hash}")
    source = root / "results" / SOURCE_CAMPAIGN
    prepare = json.loads((source / "prepare-complete.json").read_text(encoding="utf-8"))
    if prepare.get("ecgfix_commit") != ECGFIX_COMMIT or prepare.get("clocs_sha256") != CLOCS_SHA256:
        raise RuntimeError("source embedding manifest does not match the pinned encoder")
    for dataset in DATASETS:
        for split in ("train", "val", "test"):
            folder = source / "shared" / "embeddings" / dataset / split
            for name in ("CLOCS.npy", "CLOCS_y.npy", "CLOCS_done.json"):
                if not (folder / name).is_file():
                    raise RuntimeError(f"missing frozen embedding artifact: {folder / name}")
    return {
        "source_campaign": SOURCE_CAMPAIGN,
        "ecgfix_commit": ECGFIX_COMMIT,
        "clocs_sha256": checkpoint_hash,
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_views(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim != 3:
        raise ValueError(f"expected CLOCS embeddings with 3 dimensions, got {values.shape}")
    if values.shape[1] == 2:
        return values[:, 0, :], values[:, 1, :]
    if values.shape[2] == 2:
        return values[:, :, 0], values[:, :, 1]
    raise ValueError(f"cannot identify CLOCS view axis in {values.shape}")


def load_split(source: Path, dataset: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    folder = source / "shared" / "embeddings" / dataset / split
    x = np.load(folder / "CLOCS.npy").astype(np.float32, copy=False)
    y = (np.load(folder / "CLOCS_y.npy") > 0).astype(np.float32, copy=False)
    return x, y


def macro_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float, list[dict]]:
    rows = []
    aucs, auprcs = [], []
    for index in range(y_true.shape[1]):
        truth = y_true[:, index]
        if np.unique(truth).size < 2:
            auc = None
        else:
            auc = float(roc_auc_score(truth, y_prob[:, index]))
            aucs.append(auc)
        auprc = float(average_precision_score(truth, y_prob[:, index]))
        auprcs.append(auprc)
        rows.append({"class_index": index, "auroc": auc, "auprc": auprc,
                     "positives": int(truth.sum()), "total": int(truth.size)})
    return float(np.mean(aucs)), float(np.mean(auprcs)), rows


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    view1, view2 = split_views(x)
    outputs = []
    model.eval()
    for start in range(0, len(view1), batch_size):
        a = torch.from_numpy(view1[start:start + batch_size]).to(device)
        b = torch.from_numpy(view2[start:start + batch_size]).to(device)
        prob = 0.5 * (torch.sigmoid(model(a)) + torch.sigmoid(model(b)))
        outputs.append(prob.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def train_unit(root: Path, dataset: str, fraction: float, seed: int, device: torch.device,
               audit: dict) -> dict:
    source = root / "results" / SOURCE_CAMPAIGN
    output = root / "results" / CAMPAIGN / f"seed-{seed}" / dataset / f"{fraction:g}"
    done = output / "complete.json"
    if done.is_file():
        value = json.loads(done.read_text(encoding="utf-8"))
        if value.get("state") == "complete" and value.get("protocol") == "pytorch-linear-v1":
            return value

    seed_everything(seed)
    x_train, y_train = load_split(source, dataset, "train")
    x_val, y_val = load_split(source, dataset, "val")
    x_test, y_test = load_split(source, dataset, "test")

    generator = torch.Generator().manual_seed(seed)
    count = max(1, int(len(x_train) * fraction))
    indices = torch.randperm(len(x_train), generator=generator)[:count].numpy()
    train1, train2 = split_views(x_train[indices])
    train_x = np.concatenate((train1, train2), axis=0)
    train_y = np.concatenate((y_train[indices], y_train[indices]), axis=0)

    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    train_x = (train_x - mean) / std
    val1, val2 = split_views(x_val)
    x_val_scaled = np.stack(((val1 - mean) / std, (val2 - mean) / std), axis=2)
    test1, test2 = split_views(x_test)
    x_test_scaled = np.stack(((test1 - mean) / std, (test2 - mean) / std), axis=2)

    dataset_train = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, generator=generator,
                        num_workers=0, pin_memory=True)
    model = nn.Linear(train_x.shape[1], train_y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -float("inf")
    best_epoch = 0
    best_state = None
    history = []
    stale = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        examples = 0
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(features)
            examples += len(features)
        val_prob = predict(model, x_val_scaled, device)
        val_auc, val_auprc, _ = macro_metrics(y_val, val_prob)
        record = {"epoch": epoch, "train_loss": loss_sum / examples,
                  "val_macro_auroc": val_auc, "val_macro_auprc": val_auprc}
        history.append(record)
        atomic_json(output / "status.json", {
            "state": "training", "dataset": dataset, "fraction": fraction, "seed": seed,
            "epoch": epoch, "best_epoch": best_epoch, "best_val_macro_auroc": best_auc,
            "device": str(device), **audit,
        })
        if val_auc > best_auc + 1e-8:
            best_auc = val_auc
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= MIN_EPOCHS and stale >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("linear probe did not produce a checkpoint")
    model.load_state_dict(best_state)
    test_prob = predict(model, x_test_scaled, device)
    test_auc, test_auprc, per_class = macro_metrics(y_test, test_prob)

    output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "mean": mean, "std": std}, output / "best.pt")
    np.savez_compressed(output / "predictions.npz", y_true=y_test, y_prob=test_prob)
    atomic_json(output / "history.json", history)
    result = {
        "state": "complete", "protocol": "pytorch-linear-v1", "dataset": dataset,
        "fraction": fraction, "seed": seed, "train_records": int(count),
        "best_epoch": best_epoch, "best_val_macro_auroc": best_auc,
        "test_macro_auroc": test_auc, "test_macro_auprc": test_auprc,
        "per_class": per_class, "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS, "early_stopping_patience": PATIENCE,
        "selection_metric": "validation_macro_auroc", "device": str(device), **audit,
    }
    atomic_json(done, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("PyTorch GPU linear probe requires a visible CUDA device")
    audit = verify(root)
    campaign = root / "results" / CAMPAIGN
    status = campaign / f"seed-{args.seed}-status.json"
    completed = 0
    for dataset in DATASETS:
        for fraction in FRACTIONS:
            atomic_json(status, {"state": "running", "seed": args.seed, "dataset": dataset,
                                 "fraction": fraction, "completed_units": completed,
                                 "total_units": 18, "device": str(device), **audit})
            train_unit(root, dataset, fraction, args.seed, device, audit)
            completed += 1
    complete = {"state": "complete", "seed": args.seed, "completed_units": completed,
                "total_units": 18, "device": str(device), **audit}
    atomic_json(campaign / f"seed-{args.seed}-complete.json", complete)
    atomic_json(status, complete)


if __name__ == "__main__":
    main()
