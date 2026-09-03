"""Build the campaign-wide gate from 60 downloaded train/validation manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol import CAMPAIGN, SEEDS, TASKS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = {(task, seed) for task in TASKS for seed in SEEDS}
    rows = {}
    hashes = {"hug": set(), "deepsets": set(), "hilar": set()}
    for path in args.manifests.rglob("train-val-complete.json"):
        row = json.loads(path.read_text())
        key = (row.get("task"), row.get("seed"))
        if key not in expected or key in rows:
            raise RuntimeError(f"unexpected or duplicate manifest {key}: {path}")
        checks = [
            row.get("status") == "complete", row.get("selection_metric") == "macro_auroc",
            row.get("anchor_lambda") == 0.0, row.get("test_used_for_selection") is False,
            row.get("feature_checkpoint_exact") is True,
        ]
        if not all(checks):
            raise RuntimeError(f"provenance gate failed for {key}: {row}")
        for model in hashes:
            value = row.get(f"{model}_checkpoint_sha256", "")
            if len(value) != 64:
                raise RuntimeError(f"bad {model} hash for {key}")
            hashes[model].add(value)
        rows[key] = row
    if set(rows) != expected:
        raise RuntimeError(f"missing manifests: {sorted(expected - set(rows))}")
    if any(len(values) != 60 for values in hashes.values()):
        raise RuntimeError({model: len(values) for model, values in hashes.items()})
    payload = {
        "schema_version": 1, "status": "passed", "campaign": CAMPAIGN,
        "train_val_units": 60, "formal_test_authorized": True,
        "selection_metric": "macro_auroc", "test_used_for_selection": False,
        "unique_checkpoint_hashes": {model: len(values) for model, values in hashes.items()},
        "runs": {f"{task}-seed{seed}": rows[(task, seed)] for task, seed in sorted(expected)},
    }
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
