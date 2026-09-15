#!/usr/bin/env python3
"""Compare CLEAR-HUG, HeartLang, and HiLAR fixed-threshold F1."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

FRACTIONS = ("100pct", "10pct", "1pct")
TASKS = ("superdiagnostic", "subdiagnostic", "form", "rhythm", "cpsc2018", "csn")
SEEDS = (42, 46, 55)


def f1(truth: np.ndarray, score: np.ndarray, threshold: float) -> tuple[float, float]:
    actual = truth.astype(bool)
    predicted = score >= threshold
    tp = np.sum(actual & predicted, axis=0, dtype=np.int64)
    fp = np.sum(~actual & predicted, axis=0, dtype=np.int64)
    fn = np.sum(actual & ~predicted, axis=0, dtype=np.int64)
    denominator = 2 * tp + fp + fn
    class_scores = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=np.float64), where=denominator != 0)
    micro_denominator = int(2 * tp.sum() + fp.sum() + fn.sum())
    return float(class_scores.mean()), float(2 * tp.sum() / micro_denominator) if micro_denominator else 0.0


def comparative_path(workspace: Path, fraction: str, task: str, seed: int, model: str) -> Path:
    leaf = "hug" if model == "clear_hug" else "paired"
    if seed in (42, 46):
        if fraction == "100pct":
            return workspace / "results/q1-clear-deepsets-hilar-10seed-20260903/assembled" / f"{task}-seed{seed}/formal-test/{leaf}/test_predictions.npz"
        return workspace / "results/q1-six-task-low-label-10seed-20260909" / fraction / f"{task}-seed{seed}/formal-test/{leaf}/test_predictions.npz"
    return workspace / "results/q1-six-task-three-fraction-seeds52-61-20260910" / fraction / f"{task}-seed{seed}/formal-test/{leaf}/test_predictions.npz"


def load_prediction(workspace: Path, fraction: str, task: str, seed: int, model: str):
    if model == "heartlang":
        path = workspace / "src/CLEAR-HUG/results/q1-heartlang-stmem-linear-probe-seeds42-46-55-20260914" / fraction / f"{task}-seed{seed}/heartlang/formal-test/test_predictions.npz"
        score_key = "probabilities"
    else:
        path = comparative_path(workspace, fraction, task, seed, model)
        score_key = "hug_score" if model == "clear_hug" else "direct_score"
    if not path.exists():
        raise RuntimeError(f"missing prediction: {path}")
    with np.load(path, allow_pickle=True) as values:
        return np.asarray(values["y_true"]), np.asarray(values[score_key]), path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    rows = []
    for fraction in FRACTIONS:
        for task in TASKS:
            for seed in SEEDS:
                reference = None
                for model in ("clear_hug", "heartlang", "hilar"):
                    truth, score, path = load_prediction(args.workspace, fraction, task, seed, model)
                    if reference is None:
                        reference = truth
                    elif not np.array_equal(truth, reference):
                        raise RuntimeError(f"label/order mismatch: {fraction}:{task}:{seed}:{model}")
                    macro, micro = f1(truth, score, args.threshold)
                    rows.append({"fraction": fraction, "task": task, "seed": seed, "model": model,
                                 "threshold": args.threshold, "macro_f1": macro, "micro_f1": micro,
                                 "prediction": str(path.resolve())})
    summary = []
    for fraction in FRACTIONS:
        for task in TASKS:
            for model in ("clear_hug", "heartlang", "hilar"):
                group = [row for row in rows if row["fraction"] == fraction and row["task"] == task and row["model"] == model]
                for metric in ("macro_f1", "micro_f1"):
                    values = np.asarray([row[metric] for row in group])
                    summary.append({"fraction": fraction, "task": task, "model": model, "metric": metric,
                                    "mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)),
                                    "values": {str(row["seed"]): row[metric] for row in group}})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "seed-results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps({
        "models": ["CLEAR-HUG", "HeartLang", "no-anchor HiLAR"], "seeds": list(SEEDS),
        "threshold": args.threshold, "threshold_selected_on_test": False,
        "label_and_order_audit": "passed", "seed_results": rows, "summary": summary,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"seed_rows": len(rows), "groups": len(summary), "label_and_order_audit": "passed"}))


if __name__ == "__main__":
    main()
