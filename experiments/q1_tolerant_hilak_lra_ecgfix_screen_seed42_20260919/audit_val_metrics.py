#!/usr/bin/env python3
"""Audit thresholded validation metrics for all three locked model stages."""
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
def infer(model: nn.Module, dataset: FeatureDataset, device: torch.device,
          stage: str) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0,
                        pin_memory=True)
    logits = []
    model.eval()
    for global_f, lead_f, local_f, local_mask, _ in loader:
        global_f = global_f.to(device, non_blocking=True)
        if stage == "tolerant_ecg":
            output = model(global_f)
        elif stage == "hila_k":
            output = model(global_f, lead_f.to(device, non_blocking=True))
        elif stage == "hila_k_lra":
            output = model(
                global_f,
                lead_f.to(device, non_blocking=True),
                local_f.to(device, non_blocking=True),
                local_mask.to(device, non_blocking=True),
            )
        else:
            raise ValueError(stage)
        logits.append(output.cpu().numpy())
    logits_array = np.concatenate(logits).astype(np.float64)
    probabilities = 1.0 / (1.0 + np.exp(-logits_array))
    return logits_array, probabilities


def class_diagnostics(labels: np.ndarray, predictions: np.ndarray) -> list[dict]:
    values = []
    for index in range(labels.shape[1]):
        truth, pred = labels[:, index].astype(bool), predictions[:, index].astype(bool)
        tp = int(np.sum(truth & pred))
        fp = int(np.sum(~truth & pred))
        fn = int(np.sum(truth & ~pred))
        tn = int(np.sum(~truth & ~pred))
        denominator = 2 * tp + fp + fn
        values.append({
            "class_index": index,
            "positives": int(np.sum(truth)),
            "prevalence": float(np.mean(truth)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "f1_manual": float(2 * tp / denominator) if denominator else 0.0,
        })
    return values


def audit_metrics(labels: np.ndarray, logits: np.ndarray,
                  probabilities: np.ndarray) -> dict:
    predictions = probabilities >= 0.5
    logits_predictions = logits >= 0.0
    per_class = class_diagnostics(labels, predictions)
    manual_macro_f1 = float(np.mean([value["f1_manual"] for value in per_class]))
    sklearn_macro_f1 = float(
        f1_score(labels, predictions, average="macro", zero_division=0)
    )
    manual_exact = float(np.mean(np.all(predictions == labels.astype(bool), axis=1)))
    sklearn_exact = float(accuracy_score(labels, predictions))
    macro_auroc, macro_auprc = macro_metrics(labels, probabilities)
    return {
        "probability_mean": float(np.mean(probabilities)),
        "probability_std": float(np.std(probabilities)),
        "true_labels_per_record": float(np.mean(np.sum(labels, axis=1))),
        "predicted_labels_per_record": float(np.mean(np.sum(predictions, axis=1))),
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
        "macro_f1_manual": manual_macro_f1,
        "macro_f1_sklearn": sklearn_macro_f1,
        "macro_f1_absolute_difference": abs(manual_macro_f1 - sklearn_macro_f1),
        "exact_match_accuracy_manual": manual_exact,
        "exact_match_accuracy_sklearn": sklearn_exact,
        "exact_match_absolute_difference": abs(manual_exact - sklearn_exact),
        "hamming_accuracy": float(np.mean(predictions == labels.astype(bool))),
        "probability_threshold_equals_logit_threshold": bool(
            np.array_equal(predictions, logits_predictions)
        ),
        "probabilities_exactly_at_0_5": int(np.sum(probabilities == 0.5)),
        "zero_positive_classes": int(np.sum(np.sum(labels, axis=0) == 0)),
        "zero_division_policy": "class F1 is 0 when 2TP+FP+FN is 0",
        "per_class": per_class,
    }


def manifest_audit(root: Path, task: str) -> dict:
    folders = {
        "baseline": root / "results" / BASELINE_CAMPAIGN / "shared/embeddings/TolerantECG" / task / "val",
        "hila_k": root / "results" / HILAK_CAMPAIGN / "shared/masked_features" / task / "val",
        "lra": root / "results" / CAMPAIGN / "shared/local_features" / task / "val",
    }
    manifests = {
        name: json.loads((folder / "complete.json").read_text(encoding="utf-8"))
        for name, folder in folders.items()
    }
    hashes = {
        "source_indices_sha256": {
            name: value["source_indices_sha256"] for name, value in manifests.items()
        },
        "source_labels_sha256": {
            name: value.get("source_labels_sha256", value.get("labels_sha256"))
            for name, value in manifests.items()
        },
    }
    return {
        "records": {name: value["records"] for name, value in manifests.items()},
        "hashes": hashes,
        "record_counts_match": len({value["records"] for value in manifests.values()}) == 1,
        "source_indices_match": len(set(hashes["source_indices_sha256"].values())) == 1,
        "source_labels_match": len(set(hashes["source_labels_sha256"].values())) == 1,
        "class_order_evidence": "identical source label-array SHA256 across all three feature paths",
    }


def audit_task(root: Path, task: str, device: torch.device) -> dict:
    values = load_split(root, task, "val")
    val_global, val_lead, val_local, val_mask, val_labels, _, _ = values
    baseline_state, hilak_state, mean, std, _ = checkpoints(root, task)
    dataset = FeatureDataset(
        val_global, val_lead, val_local, val_mask, val_labels, mean, std
    )
    labels = np.asarray(val_labels, dtype=np.int32)
    classes = labels.shape[1]
    output = root / "results" / CAMPAIGN / task
    final_checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)

    baseline = nn.Linear(baseline_state["weight"].shape[1], classes)
    baseline.load_state_dict(baseline_state, strict=True)
    hila = HILAK(classes, baseline_state)
    hila.load_state_dict(hilak_state, strict=True)
    final = HILAKLRA(classes, baseline_state, hilak_state)
    final.load_state_dict(final_checkpoint["state_dict"], strict=True)

    models = {}
    for stage, model in (
        ("tolerant_ecg", baseline), ("hila_k", hila), ("hila_k_lra", final)
    ):
        logits, probabilities = infer(model.to(device), dataset, device, stage)
        models[stage] = audit_metrics(labels, logits, probabilities)
    result = {
        "task": task,
        "split": "validation",
        "seed": SEED,
        "fraction": FRACTION,
        "threshold": "sigmoid(logit) >= 0.5 (equivalent to logit >= 0.0)",
        "manifest_audit": manifest_audit(root, task),
        "models": models,
        "test_data_loaded": False,
        "test_evaluations": 0,
    }
    atomic_json(output / "val_metric_audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    root = args.root.resolve()
    tasks = [audit_task(root, task, device) for task in TASKS]
    summary = {
        "state": "complete",
        "split": "validation",
        "seed": SEED,
        "fraction": FRACTION,
        "tasks": tasks,
        "all_manifest_checks_pass": all(
            task["manifest_audit"][key]
            for task in tasks
            for key in ("record_counts_match", "source_indices_match", "source_labels_match")
        ),
        "all_threshold_equivalence_checks_pass": all(
            model["probability_threshold_equals_logit_threshold"]
            for task in tasks for model in task["models"].values()
        ),
        "maximum_f1_crosscheck_difference": max(
            model["macro_f1_absolute_difference"]
            for task in tasks for model in task["models"].values()
        ),
        "maximum_exact_match_crosscheck_difference": max(
            model["exact_match_absolute_difference"]
            for task in tasks for model in task["models"].values()
        ),
        "test_data_loaded": False,
        "test_evaluations": 0,
    }
    result_root = root / "results" / CAMPAIGN
    atomic_json(result_root / "val_metric_audit_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
