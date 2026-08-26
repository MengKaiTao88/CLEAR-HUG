"""Shared validation-only data and metric utilities for the Q1 campaign."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import Dataset


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_development_paths(*paths: Path) -> None:
    """Reject paths that could silently expose a formal test split."""

    for path in paths:
        lowered = {part.lower() for part in Path(path).parts}
        if "test" in lowered:
            raise RuntimeError(f"development-only campaign refuses test path: {path}")


def macro_metrics(truth: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(truth)
    score = np.asarray(score)
    valid = [i for i in range(truth.shape[1]) if np.unique(truth[:, i]).size == 2]
    if not valid:
        raise RuntimeError("no validation class has both positive and negative examples")
    return {
        "macro_auroc": float(roc_auc_score(truth[:, valid], score[:, valid], average="macro")),
        "macro_auprc": float(
            average_precision_score(truth[:, valid], score[:, valid], average="macro")
        ),
        "valid_classes": len(valid),
    }


class CachedFeatureDataset(Dataset):
    """Memory-mapped local features produced by ``cache_clear_features.py``."""

    def __init__(self, root: Path):
        ensure_development_paths(root)
        required = ("local.npy", "valid.npy", "baseline_logits.npy", "labels.npy")
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{root}: missing {missing}")
        self.local = np.load(root / "local.npy", mmap_mode="r")
        self.valid = np.load(root / "valid.npy", mmap_mode="r")
        self.logits = np.load(root / "baseline_logits.npy", mmap_mode="r")
        self.labels = np.load(root / "labels.npy", mmap_mode="r")
        sizes = {len(self.local), len(self.valid), len(self.logits), len(self.labels)}
        if len(sizes) != 1:
            raise RuntimeError(f"{root}: cached arrays have inconsistent lengths")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            np.asarray(self.local[index], dtype=np.float32),
            np.asarray(self.valid[index], dtype=np.bool_),
            np.asarray(self.logits[index], dtype=np.float32),
            np.asarray(self.labels[index], dtype=np.float32),
        )


def baseline_from_dataset(dataset: CachedFeatureDataset) -> dict[str, float | int]:
    score = 1.0 / (1.0 + np.exp(-np.asarray(dataset.logits, dtype=np.float32)))
    return macro_metrics(np.asarray(dataset.labels), score)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = path.with_name(path.name + ".incoming")
    incoming.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(incoming, path)
