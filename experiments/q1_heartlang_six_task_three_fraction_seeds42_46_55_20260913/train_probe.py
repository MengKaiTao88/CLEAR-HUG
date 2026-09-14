#!/usr/bin/env python3
"""Train the shared deterministic multilabel linear probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
# Required by PyTorch for deterministic CuBLAS GEMM on CUDA >= 10.2.  Set it
# before the first CUDA context is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.utils.data import DataLoader, TensorDataset

from common import array_sha256, atomic_json, metrics, sha256
from protocol import CAMPAIGN, FRACTIONS, MODELS, SEEDS, TASKS


def head_hash(head: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(head.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def evaluate(head, features, labels, batch_size=1024):
    scores = []
    for start in range(0, len(labels), batch_size):
        logits = head(features[start:start + batch_size].cuda(non_blocking=True))
        scores.append(torch.sigmoid(logits).cpu().numpy())
    return metrics(labels.numpy(), np.concatenate(scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--fraction", choices=tuple(FRACTIONS), required=True)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    complete = args.output / "training-complete.json"
    if complete.exists():
        return
    if args.output.exists():
        raise RuntimeError(f"refusing incomplete existing output {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    train_root = args.features / args.model / args.task / "train"
    val_root = args.features / args.model / args.task / "val"
    for root in (train_root, val_root):
        manifest = json.loads((root / "feature-manifest.json").read_text())
        if manifest.get("status") != "complete" or manifest.get("encoder_frozen") is not True:
            raise RuntimeError(f"invalid feature manifest {root}")
    x_train_all = np.load(train_root / "features.npy", mmap_mode="r")
    y_train_all = np.load(train_root / "labels.npy", mmap_mode="r")
    x_val = torch.from_numpy(np.asarray(np.load(val_root / "features.npy"), dtype=np.float32))
    y_val = torch.from_numpy(np.asarray(np.load(val_root / "labels.npy"), dtype=np.float32))
    classes, _, _, total = TASKS[args.task]
    if len(x_train_all) != total or y_train_all.shape != (total, classes):
        raise RuntimeError("unexpected frozen training feature shape")
    fraction = FRACTIONS[args.fraction]
    if fraction == 1.0:
        indices = np.arange(total, dtype=np.int64)
    else:
        indices = np.asarray(random.Random(args.seed).sample(range(total), int(total * fraction)), dtype=np.int64)
    x_train = torch.from_numpy(np.asarray(x_train_all[indices], dtype=np.float32))
    y_train = torch.from_numpy(np.asarray(y_train_all[indices], dtype=np.float32))
    batch_size = 256
    while batch_size > len(indices) and batch_size > 1:
        batch_size //= 2
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    head = torch.nn.Linear(768, classes).cuda()
    initial_hash = head_hash(head)
    optimizer = torch.optim.AdamW(head.parameters(), lr=5e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05)
    criterion = torch.nn.BCEWithLogitsLoss()
    steps_per_epoch = len(indices) // batch_size
    if steps_per_epoch < 1:
        raise RuntimeError("empty training epoch")
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = 10 * steps_per_epoch

    def learning_rate(step):
        if step < warmup_steps:
            ratio = step / max(1, warmup_steps - 1)
            return 1e-6 + ratio * (5e-3 - 1e-6)
        ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
        return 1e-5 + 0.5 * (5e-3 - 1e-5) * (1 + math.cos(math.pi * ratio))

    args.output.mkdir(parents=True)
    subset_payload = {
        "task": args.task, "fraction_key": args.fraction, "fraction": fraction,
        "seed": args.seed, "source_records": total, "selected_records": len(indices),
        "indices_sha256": array_sha256(indices),
        "selected_labels_sha256": array_sha256(np.asarray(y_train_all[indices], dtype=np.float32)),
    }
    atomic_json(args.output / "subset-manifest.json", subset_payload)
    best = -float("inf")
    best_epoch = None
    global_step = 0
    log_path = args.output / "log.txt"
    dataset = TensorDataset(x_train, y_train)
    for epoch in range(1, args.epochs + 1):
        generator = torch.Generator().manual_seed(args.seed * 1000 + epoch)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, drop_last=True, num_workers=0)
        head.train()
        losses = []
        for features, labels in loader:
            lr = learning_rate(global_step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            logits = head(features.cuda(non_blocking=True))
            loss = criterion(logits, labels.cuda(non_blocking=True))
            loss.backward(); optimizer.step()
            losses.append(float(loss.detach().cpu()))
            global_step += 1
        head.eval()
        validation = evaluate(head, x_val, y_val)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "lr": lr,
               "val_roc_auc": validation["macro_auroc"], "val_pr_auc": validation["macro_auprc"],
               "valid_classes": validation["valid_classes"]}
        with log_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        if validation["macro_auroc"] > best:
            best = validation["macro_auroc"]; best_epoch = epoch
            incoming = args.output / "checkpoint-best.pth.incoming"
            torch.save({"model": head.state_dict(), "epoch": epoch, "validation": validation,
                        "initial_head_sha256": initial_hash, "protocol": "frozen-encoder-linear-probe"}, incoming)
            os.replace(incoming, args.output / "checkpoint-best.pth")
    payload = {
        "status": "complete", "campaign": CAMPAIGN, "model": args.model,
        "task": args.task, "fraction_key": args.fraction, "fraction": fraction,
        "seed": args.seed, "classes": classes, "epochs": args.epochs,
        "batch_size": batch_size, "optimizer": "AdamW", "learning_rate": 5e-3,
        "weight_decay": 0.05, "warmup_epochs": 10, "min_lr": 1e-5,
        "selection_metric": "validation_macro_auroc", "test_used_for_selection": False,
        "encoder_frozen": True, "trainable_parameters": ["weight", "bias"],
        "initial_head_sha256": initial_hash, "best_epoch": best_epoch,
        "best_validation_auroc": best,
        "checkpoint_sha256": sha256(args.output / "checkpoint-best.pth"),
        "subset_indices_sha256": subset_payload["indices_sha256"],
    }
    atomic_json(complete, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
