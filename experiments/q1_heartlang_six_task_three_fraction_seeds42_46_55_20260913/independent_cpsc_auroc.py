#!/usr/bin/env python3
"""Independently recompute CPSC seed-42 ST-MEM AUROC with scikit-learn."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with np.load(args.prediction, allow_pickle=True) as values:
        truth = np.asarray(values["y_true"], dtype=np.float32)
        if "probabilities" in values:
            probability = np.asarray(values["probabilities"], dtype=np.float64)
        elif "logits" in values:
            logits = np.asarray(values["logits"], dtype=np.float64)
            probability = 1.0 / (1.0 + np.exp(-logits))
        else:
            raise RuntimeError("prediction has neither probabilities nor logits")
    per_class = [float(roc_auc_score(truth[:, index], probability[:, index])) for index in range(truth.shape[1])]
    report = {
        "source_prediction": str(args.prediction.resolve()),
        "implementation": "sklearn.metrics.roc_auc_score",
        "sklearn_call": "roc_auc_score(y_true, y_prob, average='macro')",
        "records": len(truth), "classes": truth.shape[1],
        "macro_auroc": float(roc_auc_score(truth, probability, average="macro")),
        "per_class_auroc": {str(index): value for index, value in enumerate(per_class)},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "y_true_and_sigmoid_probability.npz").open("wb") as handle:
        np.savez_compressed(handle, y_true=truth, sigmoid_probability=probability)
    (args.output / "independent-auroc.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
