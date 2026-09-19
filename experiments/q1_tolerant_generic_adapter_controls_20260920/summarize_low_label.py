#!/usr/bin/env python3
"""Summarize 1%/10% validation controls against locked LP and HiLAR."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from protocol import BASELINE_CAMPAIGN, LOW_LABEL_CAMPAIGN, METHODS, SEEDS, TASKS

HILAR_CAMPAIGN = "q1-tolerant-hilar-ecgfix-low-label-formal-20260919"
FRACTIONS = (0.01, 0.1)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve(); rows = []
    for fraction in FRACTIONS:
        for seed in SEEDS:
            for task in TASKS:
                baseline = read(root / "results" / BASELINE_CAMPAIGN / "TolerantECG"
                                / f"seed-{seed}" / task / f"{fraction:g}" / "complete.json")
                hilar = read(root / "results" / HILAR_CAMPAIGN / f"fraction-{fraction:g}"
                             / f"seed-{seed}" / task / "hila-k-lra" / "complete.json")
                values = {"lp": baseline["best_val_macro_auroc"],
                          "hilar": hilar["best_val_macro_auroc"]}
                metadata = {}
                baseline_indices = [int(value) for value in baseline["train_subset_indices"]]
                for method in METHODS:
                    result = read(root / "results" / LOW_LABEL_CAMPAIGN / f"seed-{seed}"
                                  / task / f"{fraction:g}" / method / "complete.json")
                    values[method] = result["best_val_macro_auroc"]
                    if result["train_records"] != len(baseline_indices):
                        raise RuntimeError(f"subset size mismatch for {fraction}/{seed}/{task}")
                    metadata[method] = {key: result[key] for key in
                        ("train_records", "train_subset_indices_sha256",
                         "trainable_parameters", "hilar_parameter_budget",
                         "parameter_mismatch_percent", "hidden_width",
                         "best_epoch", "epochs_ran")}
                rows.append({"fraction": fraction, "seed": seed, "task": task,
                             **values, "metadata": metadata})
    summary = {"campaign": LOW_LABEL_CAMPAIGN,
               "scope": "1%/10% labels, validation only", "rows": rows,
               "aggregate_by_fraction": {}}
    for fraction in FRACTIONS:
        selected = [row for row in rows if row["fraction"] == fraction]
        aggregate = {}
        for method in ("lp", *METHODS, "hilar"):
            values = np.asarray([row[method] for row in selected])
            aggregate[method] = {"mean_auroc": float(values.mean()),
                "sd_auroc": float(values.std(ddof=1)),
                "mean_gain_vs_lp": float(np.mean([row[method] - row["lp"]
                                                   for row in selected])),
                "wins_vs_lp": int(sum(row[method] > row["lp"] for row in selected)),
                "pairs": len(selected)}
        summary["aggregate_by_fraction"][f"{fraction:g}"] = aggregate
    output = root / "results" / LOW_LABEL_CAMPAIGN / "validation_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["aggregate_by_fraction"], indent=2))


if __name__ == "__main__":
    main()
