#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol import CAMPAIGN, TASKS


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(); campaign = args.root.resolve() / "results" / CAMPAIGN
    rows = []
    for task in TASKS:
        path = campaign / task / "complete.json"
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            rows.append({key: value[key] for key in (
                "task", "hilak_val_macro_auroc", "best_val_macro_auroc",
                "val_auroc_delta_vs_hilak", "best_epoch", "epochs_ran")})
    payload = {"campaign": CAMPAIGN, "completed": len(rows), "total": len(TASKS),
               "test_evaluations": 0, "results": rows}
    if rows:
        payload["mean_val_auroc_delta_vs_hilak"] = sum(
            row["val_auroc_delta_vs_hilak"] for row in rows) / len(rows)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
