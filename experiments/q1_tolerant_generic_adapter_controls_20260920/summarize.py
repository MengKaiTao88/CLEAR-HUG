#!/usr/bin/env python3
"""Summarize validation-only generic controls alongside locked LP and HiLAR."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from protocol import (BASELINE_CAMPAIGN, CAMPAIGN, HILAR_CAMPAIGN, METHODS,
                      SEED42_HILAK_CAMPAIGN, SEED42_LRA_CAMPAIGN, SEEDS, TASKS)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def hilar_result(root: Path, task: str, seed: int) -> dict:
    if seed == 42:
        folder = root / "results" / SEED42_LRA_CAMPAIGN / task / "complete.json"
    else:
        folder = (root / "results" / HILAR_CAMPAIGN / f"seed-{seed}" / task
                  / "hila-k-lra" / "complete.json")
    return read(folder)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve()
    rows = []
    for seed in SEEDS:
        for task in TASKS:
            baseline = read(root / "results" / BASELINE_CAMPAIGN / "TolerantECG"
                            / f"seed-{seed}" / task / "1" / "complete.json")
            values = {"lp": baseline["best_val_macro_auroc"],
                      "hilar": hilar_result(root, task, seed)["best_val_macro_auroc"]}
            metadata = {}
            for method in METHODS:
                result = read(root / "results" / CAMPAIGN / f"seed-{seed}" / task
                              / method / "complete.json")
                values[method] = result["best_val_macro_auroc"]
                metadata[method] = {key: result[key] for key in
                                    ("trainable_parameters", "hilar_parameter_budget",
                                     "parameter_mismatch_percent", "hidden_width",
                                     "best_epoch", "epochs_ran")}
            rows.append({"seed": seed, "task": task, **values, "metadata": metadata})
    summary = {"campaign": CAMPAIGN, "scope": "100% labels, validation only",
               "rows": rows, "aggregate": {}}
    for method in ("lp", *METHODS, "hilar"):
        array = np.asarray([row[method] for row in rows])
        summary["aggregate"][method] = {"mean_auroc": float(array.mean()),
            "sd_auroc": float(array.std(ddof=1)),
            "mean_gain_vs_lp": float(np.mean([row[method] - row["lp"] for row in rows])),
            "wins_vs_lp": int(sum(row[method] > row["lp"] for row in rows)), "pairs": len(rows)}
    out = root / "results" / CAMPAIGN / "validation_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
