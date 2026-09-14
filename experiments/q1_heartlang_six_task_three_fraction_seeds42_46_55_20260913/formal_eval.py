#!/usr/bin/env python3
"""Formal-test evaluator for a gated frozen-feature linear probe."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from common import atomic_json, metrics, sha256
from protocol import CAMPAIGN, FRACTIONS, MODELS, SEEDS, TASKS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--fraction", choices=tuple(FRACTIONS), required=True)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    if args.output.exists():
        if (args.output / "formal-test-result.json").exists():
            return
        raise RuntimeError(f"refusing incomplete output {args.output}")
    gate = json.loads(args.gate.read_text())
    if gate.get("status") != "passed" or gate.get("formal_test_authorized") is not True:
        raise RuntimeError("formal test not authorized")
    test_root = args.features / args.model / args.task / "test"
    manifest = json.loads((test_root / "feature-manifest.json").read_text())
    if manifest.get("test_gate") != str(args.gate) or manifest.get("status") != "complete":
        raise RuntimeError("invalid gated test feature cache")
    features = torch.from_numpy(np.asarray(np.load(test_root / "features.npy"), dtype=np.float32))
    truth = np.asarray(np.load(test_root / "labels.npy"), dtype=np.float32)
    record_ids = np.load(test_root / "record_ids.npy", allow_pickle=True)
    classes = TASKS[args.task][0]
    payload = torch.load(args.checkpoint, map_location="cpu")
    head = torch.nn.Linear(768, classes)
    incompatible = head.load_state_dict(payload["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"probe checkpoint mismatch {incompatible}")
    head.cuda().eval()
    rows = []
    with torch.inference_mode():
        for start in range(0, len(features), 1024):
            rows.append(head(features[start:start + 1024].cuda()).float().cpu().numpy())
    logits = np.concatenate(rows)
    score = torch.sigmoid(torch.from_numpy(logits.astype(np.float64))).numpy()
    result = {
        "status": "complete", "campaign": CAMPAIGN, "model": args.model,
        "task": args.task, "fraction_key": args.fraction, "seed": args.seed,
        "metrics": metrics(truth, score), "records": len(truth), "classes": classes,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
        "test_feature_sha256": manifest["features_sha256"],
        "test_used_for_selection": False,
    }
    args.output.mkdir(parents=True)
    pred = args.output / "test_predictions.npz.incoming"
    with pred.open("wb") as handle:
        np.savez_compressed(handle, y_true=truth, logits=logits, probabilities=score, record_ids=record_ids)
    os.replace(pred, args.output / "test_predictions.npz")
    result["predictions_sha256"] = sha256(args.output / "test_predictions.npz")
    atomic_json(args.output / "formal-test-result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
