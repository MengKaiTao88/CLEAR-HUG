#!/usr/bin/env python3
"""Compare TolerantECG, HILA-K, and HILA-K+LRA on validation.

This script deliberately uses only the validation feature arrays.  Macro F1 and
exact-match accuracy follow the repository's ``analyze_ecg_classification``
convention; macro AUPRC is threshold independent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from model import HILAK, HILAKLRA
from protocol import (BASELINE_CAMPAIGN, CAMPAIGN, FRACTION, HILAK_CAMPAIGN,
                      SEED, TASKS)
from train import FeatureDataset, atomic_json, checkpoints, load_split, macro_metrics


@torch.no_grad()
def predict(model: nn.Module, dataset: FeatureDataset, device: torch.device,
            stage: str) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0,
                        pin_memory=True)
    outputs = []
    model.eval()
    for global_f, lead_f, local_f, local_mask, _ in loader:
        global_f = global_f.to(device, non_blocking=True)
        if stage == "tolerant_ecg":
            logits = model(global_f)
        elif stage == "hila_k":
            logits = model(global_f, lead_f.to(device, non_blocking=True))
        elif stage == "hila_k_lra":
            logits = model(
                global_f,
                lead_f.to(device, non_blocking=True),
                local_f.to(device, non_blocking=True),
                local_mask.to(device, non_blocking=True),
            )
        else:
            raise ValueError(f"unknown stage: {stage}")
        outputs.append(torch.sigmoid(logits).cpu().numpy())
    probabilities = np.concatenate(outputs)
    if not np.isfinite(probabilities).all():
        raise RuntimeError(f"{stage} probabilities contain non-finite values")
    return probabilities


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = (probabilities > 0.5).astype(np.int32)
    macro_auroc, macro_auprc = macro_metrics(labels, probabilities)
    return {
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "exact_match_accuracy": float(accuracy_score(labels, predictions)),
        "labelwise_accuracy": float(np.mean(labels == predictions)),
        "micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
    }


def evaluate(root: Path, task: str, device: torch.device) -> dict:
    val_global, val_lead, val_local, val_mask, val_labels, _, _ = load_split(
        root, task, "val"
    )
    baseline_state, hilak_state, mean, std, hilak_complete = checkpoints(root, task)
    dataset = FeatureDataset(
        val_global, val_lead, val_local, val_mask, val_labels, mean, std
    )

    output = root / "results" / CAMPAIGN / task
    labels = np.asarray(val_labels, dtype=np.int32)
    classes = labels.shape[1]

    baseline_folder = (
        root / "results" / BASELINE_CAMPAIGN / "TolerantECG"
        / f"seed-{SEED}" / task / f"{FRACTION:g}"
    )
    baseline_complete = json.loads(
        (baseline_folder / "complete.json").read_text(encoding="utf-8")
    )
    final_complete = json.loads((output / "complete.json").read_text(encoding="utf-8"))
    final_checkpoint = torch.load(
        output / "best.pt", map_location="cpu", weights_only=True
    )

    baseline = nn.Linear(baseline_state["weight"].shape[1], classes)
    baseline.load_state_dict(baseline_state, strict=True)
    hila = HILAK(classes, baseline_state)
    hila.load_state_dict(hilak_state, strict=True)
    final = HILAKLRA(classes, baseline_state, hilak_state)
    final.load_state_dict(final_checkpoint["state_dict"], strict=True)

    models = {
        "tolerant_ecg": (baseline.to(device), baseline_complete),
        "hila_k": (hila.to(device), hilak_complete),
        "hila_k_lra": (final.to(device), final_complete),
    }
    model_results = {}
    for stage, (model, complete) in models.items():
        stage_metrics = metrics(labels, predict(model, dataset, device, stage))
        expected_auc = complete["best_val_macro_auroc"]
        if abs(stage_metrics["macro_auroc"] - expected_auc) > 1e-7:
            raise RuntimeError(
                f"{task}/{stage}: checkpoint AUROC {stage_metrics['macro_auroc']} "
                f"does not match selected result {expected_auc}"
            )
        stage_metrics["checkpoint_epoch"] = complete["best_epoch"]
        model_results[stage] = stage_metrics

    result = {
        "state": "complete",
        "task": task,
        "split": "validation",
        "seed": SEED,
        "fraction": FRACTION,
        "records": len(labels),
        "classes": labels.shape[1],
        "threshold": 0.5,
        "threshold_selection": "fixed; not optimized on validation",
        "models": model_results,
        "test_data_loaded": False,
        "test_evaluations": 0,
    }
    atomic_json(output / "val_metrics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", choices=TASKS, action="append")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    tasks = tuple(args.task) if args.task else TASKS
    results = [evaluate(args.root.resolve(), task, device) for task in tasks]
    means = {}
    metric_names = (
        "macro_auroc", "macro_auprc", "macro_f1",
        "exact_match_accuracy", "labelwise_accuracy", "micro_f1",
    )
    for stage in ("tolerant_ecg", "hila_k", "hila_k_lra"):
        means[stage] = {
            name: float(np.mean([x["models"][stage][name] for x in results]))
            for name in metric_names
        }
    summary = {
        "state": "complete",
        "split": "validation",
        "threshold": 0.5,
        "tasks": results,
        "means": means,
        "test_data_loaded": False,
        "test_evaluations": 0,
    }
    result_root = args.root.resolve() / "results" / CAMPAIGN
    atomic_json(result_root / "val_metrics_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
