#!/usr/bin/env python3
"""Evaluate each frozen validation-selected checkpoint on test exactly once."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import HILAK, HILAKLRA
from protocol import (BASELINE_CAMPAIGN, CAMPAIGN, FRACTIONS, SEEDS, TASKS,
                      TEST_FEATURE_CAMPAIGN)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = path.with_suffix(path.suffix + ".incoming")
    incoming.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(incoming, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TestFeatures(Dataset):
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


def load_test(root: Path, task: str, mean: np.ndarray, std: np.ndarray) -> TestFeatures:
    base = root / "results" / BASELINE_CAMPAIGN / "shared/embeddings/TolerantECG" / task / "test"
    lead = root / "results" / TEST_FEATURE_CAMPAIGN / "shared/masked_features" / task / "test"
    local = root / "results" / TEST_FEATURE_CAMPAIGN / "shared/local_features" / task / "test"
    manifests = [json.loads((folder / "complete.json").read_text(encoding="utf-8"))
                 for folder in (base, lead, local)]
    index_hashes = [value["source_indices_sha256"] for value in manifests]
    label_hashes = [manifests[0]["labels_sha256"], manifests[1]["source_labels_sha256"],
                    manifests[2]["source_labels_sha256"]]
    if len(set(index_hashes)) != 1 or len(set(label_hashes)) != 1:
        raise RuntimeError(f"formal-test feature provenance mismatch for {task}")
    arrays = (np.load(base / "features.npy", mmap_mode="r"),
              np.load(lead / "features.npy", mmap_mode="r"),
              np.load(local / "features.npy", mmap_mode="r"),
              np.load(local / "masks.npy", mmap_mode="r"),
              (np.load(base / "labels.npy", mmap_mode="r") > 0).astype(np.float32))
    if len({len(value) for value in arrays}) != 1:
        raise RuntimeError(f"formal-test feature length mismatch for {task}")
    return TestFeatures(*arrays, mean, std)


def checkpoint_folders(root: Path, task: str, seed: int,
                       fraction: float) -> tuple[Path, Path]:
    base = (root / "results" / CAMPAIGN / f"fraction-{fraction:g}"
            / f"seed-{seed}" / task)
    return base / "hila-k", base / "hila-k-lra"


@torch.inference_mode()
def infer(model: nn.Module, dataset: Dataset, device: torch.device, stage: str) -> np.ndarray:
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
    return {"macro_auroc": float(np.mean(aucs)), "macro_auprc": float(np.mean(aps)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "exact_match_accuracy": float(accuracy_score(labels, predictions))}


def evaluate_once(root: Path, task: str, seed: int, fraction: float,
                  device: torch.device) -> dict:
    output = (root / "results" / CAMPAIGN / "test-evaluations"
              / f"fraction-{fraction:g}" / f"seed-{seed}" / task / "complete.json")
    if output.exists():
        value = json.loads(output.read_text(encoding="utf-8"))
        if value.get("state") == "complete": return value
    baseline_folder = (root / "results" / BASELINE_CAMPAIGN / "TolerantECG"
                       / f"seed-{seed}" / task / f"{fraction:g}")
    baseline_checkpoint_path = baseline_folder / "best.pt"
    baseline_checkpoint = torch.load(baseline_checkpoint_path, map_location="cpu", weights_only=False)
    base_state = baseline_checkpoint["state_dict"]
    dataset = load_test(root, task, np.asarray(baseline_checkpoint["mean"], dtype=np.float32),
                        np.asarray(baseline_checkpoint["std"], dtype=np.float32))
    labels = np.asarray(dataset.labels, dtype=np.int32); classes = labels.shape[1]
    hila_folder, lra_folder = checkpoint_folders(root, task, seed, fraction)
    hila_path, lra_path = hila_folder / "best.pt", lra_folder / "best.pt"
    hila_checkpoint = torch.load(hila_path, map_location="cpu", weights_only=True)
    lra_checkpoint = torch.load(lra_path, map_location="cpu", weights_only=True)
    baseline_model = nn.Linear(base_state["weight"].shape[1], classes)
    baseline_model.load_state_dict(base_state, strict=True)
    hila = HILAK(classes, base_state); hila.load_state_dict(hila_checkpoint["state_dict"], strict=True)
    lra = HILAKLRA(classes, base_state, hila_checkpoint["state_dict"])
    lra.load_state_dict(lra_checkpoint["state_dict"], strict=True)
    models = {"tolerant_ecg": baseline_model, "hila_k": hila, "hila_k_lra": lra}
    result = {"state": "complete", "split": "test", "task": task, "seed": seed,
              "fraction": fraction, "threshold": 0.5, "f1_average": "macro",
              "accuracy": "exact-match/subset",
              "models": {stage: metrics(labels, infer(model.to(device), dataset, device, stage))
                         for stage, model in models.items()},
              "checkpoint_sha256": {"tolerant_ecg": file_sha256(baseline_checkpoint_path),
                                     "hila_k": file_sha256(hila_path),
                                     "hila_k_lra": file_sha256(lra_path)},
              "test_data_loaded": True, "test_evaluations": 1,
              "selection_policy": "frozen validation-selected checkpoints; no test tuning"}
    atomic_json(output, result)
    return result


def main() -> None:
    root = Path("/root/107552503710-1").resolve(); device = torch.device("cuda:0")
    rows = [evaluate_once(root, task, seed, fraction, device)
            for fraction in FRACTIONS for task in TASKS for seed in SEEDS]
    metrics_list = ("macro_auroc", "macro_auprc", "macro_f1", "exact_match_accuracy")
    methods = ("tolerant_ecg", "hila_k", "hila_k_lra")
    by_fraction = {}
    for fraction in FRACTIONS:
        fraction_rows = [row for row in rows if row["fraction"] == fraction]
        by_task = {}
        for task in TASKS:
            task_rows = [row for row in fraction_rows if row["task"] == task]
            by_task[task] = {method: {metric: {
                "mean": float(np.mean([row["models"][method][metric] for row in task_rows])),
                "sd": float(np.std([row["models"][method][metric] for row in task_rows], ddof=1)),
                "values_by_seed": {str(row["seed"]): row["models"][method][metric]
                                   for row in task_rows}}
                for metric in metrics_list} for method in methods}
        overall = {method: {metric: float(np.mean(
            [row["models"][method][metric] for row in fraction_rows]))
            for metric in metrics_list} for method in methods}
        by_fraction[f"{fraction:g}"] = {"by_task": by_task, "overall": overall}
    summary = {"state": "complete", "split": "test", "seeds": list(SEEDS),
               "fractions": list(FRACTIONS), "rows": rows, "by_fraction": by_fraction,
               "test_data_loaded": True, "test_evaluations_per_checkpoint": 1,
               "selection_policy": "frozen validation-selected checkpoints; no test tuning"}
    atomic_json(root / "results" / CAMPAIGN / "test_metrics_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
