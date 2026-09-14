#!/usr/bin/env python3
"""Audit paired predictions and summarize the three-seed formal metrics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from common import atomic_json
from protocol import FRACTIONS, MODELS, SEEDS, TASKS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    references = {}
    for fraction in FRACTIONS:
        for task in TASKS:
            for seed in SEEDS:
                for model in MODELS:
                    root = args.results / fraction / f"{task}-seed{seed}" / model / "formal-test"
                    result = json.loads((root / "formal-test-result.json").read_text())
                    with np.load(root / "test_predictions.npz", allow_pickle=True) as pred:
                        truth = pred["y_true"]; record_ids = pred["record_ids"]
                    key = task
                    if key not in references:
                        references[key] = (truth, record_ids)
                    elif not np.array_equal(truth, references[key][0]) or not np.array_equal(record_ids, references[key][1]):
                        raise RuntimeError(f"test label/order mismatch {fraction}:{task}:{seed}:{model}")
                    metric = result["metrics"]
                    rows.append({"fraction": fraction, "task": task, "seed": seed, "model": model,
                                 "macro_auroc": metric["macro_auroc"], "macro_auprc": metric["macro_auprc"]})
    if len(rows) != 108:
        raise RuntimeError(f"expected 108 rows, got {len(rows)}")
    csv_path = args.results / "seed-results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {"status": "complete", "formal_results": len(rows), "test_label_and_order_audit": "passed", "groups": {}}
    for fraction in FRACTIONS:
        summary["groups"][fraction] = {}
        for task in TASKS:
            task_summary = {"models": {}, "paired_stmem_minus_heartlang": {}}
            for model in MODELS:
                selected = [row for row in rows if row["fraction"] == fraction and row["task"] == task and row["model"] == model]
                for metric in ("macro_auroc", "macro_auprc"):
                    values = np.asarray([row[metric] for row in selected])
                    task_summary["models"].setdefault(model, {})[metric] = {
                        "mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)),
                        "values": {str(row["seed"]): row[metric] for row in selected},
                    }
            for metric in ("macro_auroc", "macro_auprc"):
                left = task_summary["models"]["heartlang"][metric]["values"]
                right = task_summary["models"]["stmem"][metric]["values"]
                deltas = {str(seed): right[str(seed)] - left[str(seed)] for seed in SEEDS}
                values = np.asarray(list(deltas.values()))
                task_summary["paired_stmem_minus_heartlang"][metric] = {
                    "mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)),
                    "positive_seeds": int((values > 0).sum()), "deltas": deltas,
                }
            summary["groups"][fraction][task] = task_summary
    atomic_json(args.results / "summary.json", summary)
    print(json.dumps({"status": "complete", "rows": 108, "audit": "passed"}, indent=2))


if __name__ == "__main__":
    main()
