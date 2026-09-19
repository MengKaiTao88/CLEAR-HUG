#!/usr/bin/env python3
"""Report AUROC, AUPRC, Macro-F1, and exact ACC for three seeds/models."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from model import HILAK, HILAKLRA
from protocol import (BASELINE_CAMPAIGN, CAMPAIGN, FRACTION, SEED42_HILAK_CAMPAIGN,
                      SEED42_LRA_CAMPAIGN, SEEDS, TASKS)
from train import Features, atomic_json, baseline, load_split


@torch.no_grad()
def infer(model: nn.Module, dataset: Features, device: torch.device, stage: str) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
    outputs = []
    model.eval()
    for global_f, lead_f, local_f, local_mask, _ in loader:
        global_f, lead_f = global_f.to(device), lead_f.to(device)
        if stage == "tolerant_ecg": logits = model(global_f)
        elif stage == "hila_k": logits = model(global_f, lead_f)
        else: logits = model(global_f, lead_f, local_f.to(device), local_mask.to(device))
        outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs)


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = probabilities >= 0.5
    aucs, aps = [], []
    for index in range(labels.shape[1]):
        truth = labels[:, index]
        if np.unique(truth).size == 2:
            aucs.append(roc_auc_score(truth, probabilities[:, index]))
        aps.append(average_precision_score(truth, probabilities[:, index]))
    return {
        "macro_auroc": float(np.mean(aucs)),
        "macro_auprc": float(np.mean(aps)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "exact_match_accuracy": float(accuracy_score(labels, predictions)),
    }


def checkpoint_folders(root: Path, task: str, seed: int) -> tuple[Path, Path]:
    if seed == 42:
        return (root / "results" / SEED42_HILAK_CAMPAIGN / task,
                root / "results" / SEED42_LRA_CAMPAIGN / task)
    base = root / "results" / CAMPAIGN / f"seed-{seed}" / task
    return base / "hila-k", base / "hila-k-lra"


def evaluate(root: Path, task: str, seed: int, device: torch.device) -> dict:
    base_state, mean, std, _ = baseline(root, task, seed)
    values = load_split(root, task, "val")
    dataset = Features(*values, mean, std)
    labels = np.asarray(dataset.labels, dtype=np.int32)
    hila_folder, lra_folder = checkpoint_folders(root, task, seed)
    hila_checkpoint = torch.load(hila_folder / "best.pt", map_location="cpu", weights_only=True)
    lra_checkpoint = torch.load(lra_folder / "best.pt", map_location="cpu", weights_only=True)
    classes = labels.shape[1]
    baseline_model = nn.Linear(base_state["weight"].shape[1], classes)
    baseline_model.load_state_dict(base_state, strict=True)
    hila = HILAK(classes, base_state); hila.load_state_dict(hila_checkpoint["state_dict"], strict=True)
    lra = HILAKLRA(classes, base_state, hila_checkpoint["state_dict"])
    lra.load_state_dict(lra_checkpoint["state_dict"], strict=True)
    models = {
        "tolerant_ecg": baseline_model,
        "hila_k": hila,
        "hila_k_lra": lra,
    }
    return {"task": task, "seed": seed, "fraction": FRACTION,
            "models": {stage: metrics(labels, infer(model.to(device), dataset, device, stage))
                       for stage, model in models.items()},
            "test_data_loaded": False, "test_evaluations": 0}


def main() -> None:
    root = Path("/root/107552503710-1").resolve()
    device = torch.device("cuda:0")
    rows = [evaluate(root, task, seed, device) for task in TASKS for seed in SEEDS]
    metric_names = ("macro_auroc", "macro_auprc", "macro_f1", "exact_match_accuracy")
    methods = ("tolerant_ecg", "hila_k", "hila_k_lra")
    by_task = {}
    for task in TASKS:
        task_rows = [row for row in rows if row["task"] == task]
        by_task[task] = {}
        for method in methods:
            by_task[task][method] = {}
            for metric in metric_names:
                values = np.asarray([row["models"][method][metric] for row in task_rows])
                by_task[task][method][metric] = {
                    "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
                    "values_by_seed": {str(row["seed"]): row["models"][method][metric]
                                       for row in task_rows},
                }
    overall = {}
    for method in methods:
        overall[method] = {
            metric: float(np.mean([row["models"][method][metric] for row in rows]))
            for metric in metric_names
        }
    result = {"state": "complete", "split": "validation", "seeds": list(SEEDS),
              "fraction": FRACTION, "threshold": 0.5,
              "f1_average": "macro", "accuracy": "exact-match/subset",
              "rows": rows, "by_task": by_task, "overall": overall,
              "test_data_loaded": False, "test_evaluations": 0}
    output = root / "results" / CAMPAIGN / "metrics_summary.json"
    atomic_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
