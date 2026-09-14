#!/usr/bin/env python3
"""Read-only integrity audit for collected HeartLang/ST-MEM predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def binary_auroc(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positives = y == 1
    n_pos = int(positives.sum())
    n_neg = len(y) - n_pos
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_ap(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="mergesort")
    ranked = y[order]
    ranked_score = score[order]
    positives = int(ranked.sum())
    cumulative = np.cumsum(ranked)
    group_ends = np.r_[np.flatnonzero(ranked_score[1:] != ranked_score[:-1]), len(ranked_score) - 1]
    true_positives = cumulative[group_ends]
    previous = np.r_[0, true_positives[:-1]]
    precision = true_positives / (group_ends + 1)
    return float(np.sum(precision * (true_positives - previous)) / positives)


def metrics(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    valid = [index for index in range(y.shape[1]) if np.unique(y[:, index]).size == 2]
    return (
        float(np.mean([binary_auroc(y[:, index], score[:, index]) for index in valid])),
        float(np.mean([binary_ap(y[:, index], score[:, index]) for index in valid])),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    files = sorted(args.root.glob("*pct/*-seed*/**/formal-test/test_predictions.npz"))
    if not files:
        raise RuntimeError(f"no predictions below {args.root}")
    references: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    failures: list[str] = []
    shuffled: dict[str, list[float]] = {}
    checked = 0
    for path in files:
        with np.load(path, allow_pickle=True) as values:
            y = values["y_true"]
            logits = values["logits"]
            probabilities = values["probabilities"]
            record_ids = values["record_ids"]
        fraction = path.parents[3].name
        task_seed = path.parents[2].name
        task = task_seed.rsplit("-seed", 1)[0]
        model = path.parents[1].name
        key = f"{fraction}:{task}"
        if key not in references:
            references[key] = (y.copy(), record_ids.copy())
        else:
            ref_y, ref_ids = references[key]
            if not np.array_equal(y, ref_y):
                failures.append(f"label/order mismatch: {path}")
            if not np.array_equal(record_ids, ref_ids):
                failures.append(f"record-id mismatch: {path}")
        if len(np.unique(record_ids)) != len(record_ids):
            failures.append(f"duplicate test record ids: {path}")
        if not np.isfinite(probabilities).all() or not np.isfinite(logits).all():
            failures.append(f"non-finite predictions: {path}")
        sigmoid = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        if not np.allclose(sigmoid, probabilities, rtol=1e-6, atol=1e-7):
            failures.append(f"probability/logit mismatch: {path}")
        auroc, auprc = metrics(y, probabilities)
        result = json.loads(path.with_name("formal-test-result.json").read_text())
        for metric_name, recomputed in (("macro_auroc", auroc), ("macro_auprc", auprc)):
            if not np.isclose(recomputed, result["metrics"][metric_name], rtol=1e-10, atol=1e-12):
                failures.append(f"{metric_name} mismatch: {path}")
        if model == "stmem" and fraction == "100pct":
            rng = np.random.default_rng(20260914)
            shuffled_y = y[rng.permutation(len(y))]
            shuffled.setdefault(task, []).append(metrics(shuffled_y, probabilities)[0])
        checked += 1
    report = {
        "status": "passed" if not failures else "failed",
        "prediction_files_checked": checked,
        "reference_test_sets": len(references),
        "labels_and_record_order_identical_within_fraction_task": not any(
            value.startswith(("label/order mismatch", "record-id mismatch")) for value in failures
        ),
        "test_record_ids_unique": not any("duplicate" in value for value in failures),
        "probabilities_match_logits": not any("probability/logit" in value for value in failures),
        "stored_metrics_exactly_recomputed": not any(value.startswith("macro_") for value in failures),
        "stmem_100pct_shuffled_label_auroc": {
            task: {"mean": float(np.mean(values)), "values": values}
            for task, values in sorted(shuffled.items())
        },
        "failures": failures,
        "scope_note": "This audit does not prove train/test waveform disjointness; training features were not downloaded.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
