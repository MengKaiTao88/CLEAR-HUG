#!/usr/bin/env python3
"""Summarize HILA-K/D/KD validation results for the three screening tasks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol import CAMPAIGN, K_CAMPAIGN, TASKS, VARIANTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    results_root = args.root.resolve() / "results"
    rows = []
    for task in TASKS:
        k = json.loads((results_root / K_CAMPAIGN / task / "complete.json").read_text(encoding="utf-8"))
        for variant in ("HILA-K",) + VARIANTS:
            path = ((results_root / K_CAMPAIGN / task / "complete.json") if variant == "HILA-K"
                    else (results_root / CAMPAIGN / variant / task / "complete.json"))
            if path.is_file():
                value = json.loads(path.read_text(encoding="utf-8"))
                rows.append({"task": task, "variant": variant,
                    "baseline_val_macro_auroc": value["baseline_val_macro_auroc"],
                    "best_val_macro_auroc": value["best_val_macro_auroc"],
                    "val_auroc_delta_vs_baseline": value["val_auroc_delta"],
                    "val_auroc_delta_vs_hila_k": value["best_val_macro_auroc"] - k["best_val_macro_auroc"],
                    "best_epoch": value["best_epoch"], "epochs_ran": value["epochs_ran"]})
    payload = {"campaign": CAMPAIGN, "tasks": list(TASKS),
               "completed_new_runs": sum(row["variant"] != "HILA-K" for row in rows),
               "total_new_runs": len(TASKS) * len(VARIANTS),
               "test_evaluations": 0, "results": rows}
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
