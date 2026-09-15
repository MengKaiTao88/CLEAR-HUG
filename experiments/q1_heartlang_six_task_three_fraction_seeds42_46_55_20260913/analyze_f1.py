#!/usr/bin/env python3
"""Compute fixed-threshold multilabel F1 from saved formal predictions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def f1_scores(truth: np.ndarray, probabilities: np.ndarray, threshold: float) -> tuple[float, float]:
    prediction = probabilities >= threshold
    positive = truth.astype(bool)
    tp = np.sum(prediction & positive, axis=0, dtype=np.int64)
    fp = np.sum(prediction & ~positive, axis=0, dtype=np.int64)
    fn = np.sum(~prediction & positive, axis=0, dtype=np.int64)
    denominator = 2 * tp + fp + fn
    per_class = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=np.float64), where=denominator != 0)
    macro = float(np.mean(per_class))
    micro_denominator = int(2 * tp.sum() + fp.sum() + fn.sum())
    micro = float(2 * tp.sum() / micro_denominator) if micro_denominator else 0.0
    return macro, micro


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.glob("*pct/*-seed*/**/formal-test/test_predictions.npz")):
        with np.load(path, allow_pickle=True) as values:
            macro, micro = f1_scores(values["y_true"], values["probabilities"], args.threshold)
        fraction = path.parents[3].name
        task_seed = path.parents[2].name
        task, seed = task_seed.rsplit("-seed", 1)
        rows.append({"fraction": fraction, "task": task, "seed": int(seed),
                     "model": path.parents[1].name, "threshold": args.threshold,
                     "macro_f1": macro, "micro_f1": micro})
    if not rows:
        raise RuntimeError(f"no formal predictions found below {args.root}")
    summary = []
    keys = sorted({(row["fraction"], row["task"], row["model"]) for row in rows})
    for fraction, task, model in keys:
        group = [row for row in rows if (row["fraction"], row["task"], row["model"]) == (fraction, task, model)]
        for metric in ("macro_f1", "micro_f1"):
            values = np.asarray([row[metric] for row in group])
            summary.append({"fraction": fraction, "task": task, "model": model,
                            "metric": metric, "mean": float(values.mean()),
                            "sample_sd": float(values.std(ddof=1)), "seeds": len(values)})
    output_csv = args.root / "seed-results-f1-threshold-0.5.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    output_json = args.root / "summary-f1-threshold-0.5.json"
    output_json.write_text(json.dumps({"threshold": args.threshold, "threshold_selected_on_test": False,
                                       "seed_results": rows, "summary": summary}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prediction_files": len(rows), "csv": str(output_csv), "json": str(output_json)}, indent=2))


if __name__ == "__main__":
    main()
