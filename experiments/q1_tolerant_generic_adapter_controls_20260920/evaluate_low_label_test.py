#!/usr/bin/env python3
"""One-time formal test of frozen 1%/10% generic-control checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import GenericResidualAdapter, ParameterMatchedMLP, trainable_parameters
from protocol import (BASELINE_CAMPAIGN, LOW_LABEL_CAMPAIGN,
                      LOW_LABEL_FORMAL_TEST_CAMPAIGN, LOW_LABEL_REFERENCE_CAMPAIGN,
                      METHODS, SEEDS, TASKS, hilar_parameter_budget)

FRACTIONS = (0.01, 0.1)


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


def indices_sha256(indices: list[int]) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


class TestFeatures(Dataset):
    def __init__(self, features, labels, mean: np.ndarray, std: np.ndarray) -> None:
        self.features, self.labels = features, labels
        self.mean, self.std = mean, std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        features = np.asarray(self.features[index], dtype=np.float32)
        return ((features - self.mean) / self.std,
                np.asarray(self.labels[index], dtype=np.float32))


def load_test(root: Path, task: str, mean: np.ndarray, std: np.ndarray,
              baseline_result: dict) -> tuple[TestFeatures, dict]:
    folder = (root / "results" / BASELINE_CAMPAIGN / "shared/embeddings/TolerantECG"
              / task / "test")
    manifest = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    if manifest["source_indices_sha256"] != baseline_result["test_source_indices_sha256"]:
        raise RuntimeError(f"test source-index provenance mismatch for {task}")
    if manifest["labels_sha256"] != baseline_result["test_labels_sha256"]:
        raise RuntimeError(f"test-label provenance mismatch for {task}")
    features = np.load(folder / "features.npy", mmap_mode="r")
    labels = (np.load(folder / "labels.npy", mmap_mode="r") > 0).astype(np.float32)
    if len(features) != len(labels):
        raise RuntimeError(f"test feature length mismatch for {task}")
    return TestFeatures(features, labels, mean, std), manifest


@torch.inference_mode()
def infer(model: nn.Module, dataset: Dataset, device: torch.device) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0,
                        pin_memory=True)
    outputs = []; model.eval()
    for features, _ in loader:
        outputs.append(torch.sigmoid(model(features.to(device, non_blocking=True))).cpu().numpy())
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


def evaluate_once(root: Path, task: str, seed: int, fraction: float, method: str,
                  device: torch.device) -> dict:
    output = (root / "results" / LOW_LABEL_FORMAL_TEST_CAMPAIGN / "evaluations"
              / f"fraction-{fraction:g}" / f"seed-{seed}" / task / method / "complete.json")
    if output.exists():
        value = json.loads(output.read_text(encoding="utf-8"))
        if value.get("state") == "complete":
            return value
        raise RuntimeError(f"refusing to overwrite incomplete formal-test result: {output}")
    baseline_folder = (root / "results" / BASELINE_CAMPAIGN / "TolerantECG"
                       / f"seed-{seed}" / task / f"{fraction:g}")
    baseline_result = json.loads((baseline_folder / "complete.json").read_text(encoding="utf-8"))
    baseline_checkpoint = torch.load(baseline_folder / "best.pt", map_location="cpu",
                                     weights_only=False)
    checkpoint_path = (root / "results" / LOW_LABEL_CAMPAIGN / f"seed-{seed}" / task
                       / f"{fraction:g}" / method / "best.pt")
    train_result = json.loads((checkpoint_path.parent / "complete.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_subset_hash = indices_sha256([int(value)
        for value in baseline_result["train_subset_indices"]])
    if len({expected_subset_hash, train_result["train_subset_indices_sha256"],
            checkpoint["train_subset_indices_sha256"]}) != 1:
        raise RuntimeError("frozen low-label subset provenance mismatch")
    if checkpoint["epoch"] != train_result["best_epoch"]:
        raise RuntimeError(f"checkpoint epoch mismatch: {checkpoint_path}")
    mean, std = (np.asarray(checkpoint["mean"], dtype=np.float32),
                 np.asarray(checkpoint["std"], dtype=np.float32))
    if not np.array_equal(mean, np.asarray(baseline_checkpoint["mean"], dtype=np.float32)):
        raise RuntimeError("checkpoint normalization mean mismatch")
    if not np.array_equal(std, np.asarray(baseline_checkpoint["std"], dtype=np.float32)):
        raise RuntimeError("checkpoint normalization std mismatch")
    dataset, manifest = load_test(root, task, mean, std, baseline_result)
    classes = dataset.labels.shape[1]
    if method == "parameter-matched-mlp":
        model = ParameterMatchedMLP(classes)
    else:
        model = GenericResidualAdapter(classes, baseline_checkpoint["state_dict"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    actual, target = trainable_parameters(model), hilar_parameter_budget(classes)
    if abs(actual - target) / target > 0.01:
        raise RuntimeError("formal-test model violates locked parameter budget")
    probabilities = infer(model.to(device), dataset, device)
    labels = np.asarray(dataset.labels, dtype=np.int32)
    result = {"state": "complete", "split": "test", "task": task, "seed": seed,
              "fraction": fraction, "method": method,
              "metrics": metrics(labels, probabilities),
              "checkpoint_sha256": file_sha256(checkpoint_path),
              "checkpoint_epoch": checkpoint["epoch"], "trainable_parameters": actual,
              "hilar_parameter_budget": target,
              "train_subset_indices_sha256": expected_subset_hash,
              "test_source_indices_sha256": manifest["source_indices_sha256"],
              "test_labels_sha256": manifest["labels_sha256"],
              "threshold": 0.5, "f1_average": "macro",
              "accuracy": "exact-match/subset", "test_data_loaded": True,
              "test_evaluations": 1,
              "selection_policy": "frozen validation-selected checkpoint; no test tuning"}
    atomic_json(output, result)
    return result


def summarize(root: Path) -> dict:
    rows = [json.loads((root / "results" / LOW_LABEL_FORMAL_TEST_CAMPAIGN / "evaluations"
            / f"fraction-{fraction:g}" / f"seed-{seed}" / task / method
            / "complete.json").read_text(encoding="utf-8"))
            for fraction in FRACTIONS for seed in SEEDS for task in TASKS for method in METHODS]
    metrics_list = ("macro_auroc", "macro_auprc", "macro_f1", "exact_match_accuracy")
    reference = json.loads((root / "results" / LOW_LABEL_REFERENCE_CAMPAIGN
                            / "test_metrics_summary.json").read_text(encoding="utf-8"))
    by_fraction = {}
    for fraction in FRACTIONS:
        selected = [row for row in rows if row["fraction"] == fraction]
        controls = {method: {metric: float(np.mean([row["metrics"][metric]
                    for row in selected if row["method"] == method]))
                    for metric in metrics_list} for method in METHODS}
        ref = reference["by_fraction"][f"{fraction:g}"]["overall"]
        by_fraction[f"{fraction:g}"] = {
            "tolerant_ecg_lp": ref["tolerant_ecg"],
            "parameter_matched_mlp": controls["parameter-matched-mlp"],
            "generic_adapter": controls["generic-adapter"],
            "hilar": ref["hila_k_lra"],
        }
    summary = {"state": "complete", "split": "test", "fractions": list(FRACTIONS),
               "seeds": list(SEEDS), "rows": rows, "publication_tables": by_fraction,
               "test_evaluations": len(rows), "test_evaluations_per_checkpoint": 1,
               "selection_policy": "frozen validation-selected checkpoints; no test tuning"}
    atomic_json(root / "results" / LOW_LABEL_FORMAL_TEST_CAMPAIGN
                / "metrics_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--total-shards", type=int, default=1)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(); root = args.root.resolve()
    if args.summarize_only:
        print(json.dumps(summarize(root)["publication_tables"], indent=2)); return
    device = torch.device(f"cuda:{args.gpu}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    index = 0
    for fraction in FRACTIONS:
        for seed in SEEDS:
            for task in TASKS:
                for method in METHODS:
                    if index % args.total_shards == args.shard:
                        print(json.dumps(evaluate_once(root, task, seed, fraction, method,
                                                       device), indent=2), flush=True)
                    index += 1


if __name__ == "__main__":
    main()

