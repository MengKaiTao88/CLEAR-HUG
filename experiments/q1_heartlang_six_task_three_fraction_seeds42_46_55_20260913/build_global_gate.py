#!/usr/bin/env python3
"""Build the global pre-test gate from collected manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import atomic_json
from protocol import CAMPAIGN, FRACTIONS, MODELS, SEEDS, TASKS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = {}
    hashes = set()
    for task in TASKS:
        for fraction in FRACTIONS:
            for seed in SEEDS:
                pair = {}
                for model in MODELS:
                    path = args.manifests / fraction / f"{task}-seed{seed}" / model / "training-complete.json"
                    payload = json.loads(path.read_text())
                    if payload.get("status") != "complete" or payload.get("test_used_for_selection") is not False:
                        raise RuntimeError(f"invalid training manifest {path}")
                    if payload.get("encoder_frozen") is not True or payload.get("selection_metric") != "validation_macro_auroc":
                        raise RuntimeError(f"protocol violation {path}")
                    checkpoint_hash = payload["checkpoint_sha256"]
                    if checkpoint_hash in hashes:
                        raise RuntimeError("duplicate checkpoint hash")
                    hashes.add(checkpoint_hash); pair[model] = payload
                if pair["heartlang"]["subset_indices_sha256"] != pair["stmem"]["subset_indices_sha256"]:
                    raise RuntimeError(f"subset mismatch {fraction}:{task}:{seed}")
                if pair["heartlang"]["initial_head_sha256"] != pair["stmem"]["initial_head_sha256"]:
                    raise RuntimeError(f"head initialization mismatch {fraction}:{task}:{seed}")
                records[f"{fraction}:{task}:{seed}"] = {
                    model: pair[model]["checkpoint_sha256"] for model in MODELS
                }
    if len(records) != 54 or len(hashes) != 108:
        raise RuntimeError("incomplete gate inputs")
    payload = {
        "status": "passed", "campaign": CAMPAIGN, "formal_test_authorized": True,
        "train_validation_units": 108, "task_seed_fraction_specs": 54,
        "selection_metric": "validation_macro_auroc", "test_used_for_selection": False,
        "paired_subset_and_head_initialization": True, "records": records,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
