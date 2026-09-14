from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = path.with_name(path.name + ".incoming")
    incoming.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(incoming, path)


def metrics(truth: np.ndarray, score: np.ndarray) -> dict:
    valid = [i for i in range(truth.shape[1]) if np.unique(truth[:, i]).size == 2]
    if not valid:
        raise RuntimeError("no valid binary label columns")
    return {
        "macro_auroc": float(roc_auc_score(truth[:, valid], score[:, valid], average="macro")),
        "macro_auprc": float(average_precision_score(truth[:, valid], score[:, valid], average="macro")),
        "valid_classes": len(valid),
        "total_classes": int(truth.shape[1]),
    }


def save_npy_atomic(path: Path, value: np.ndarray) -> None:
    incoming = path.with_name(path.name + ".incoming")
    path.parent.mkdir(parents=True, exist_ok=True)
    with incoming.open("wb") as handle:
        np.save(handle, value)
    os.replace(incoming, path)


def state_dict(payload: object) -> dict:
    if isinstance(payload, dict):
        for key in ("model", "state_dict"):
            if isinstance(payload.get(key), dict):
                return payload[key]
        if payload and all(hasattr(value, "numel") for value in payload.values()):
            return payload
    raise RuntimeError("checkpoint has no tensor state dictionary")
