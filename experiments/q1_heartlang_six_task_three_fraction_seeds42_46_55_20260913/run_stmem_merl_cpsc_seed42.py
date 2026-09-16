#!/usr/bin/env python3
"""ST-MEM CPSC seed-42 linear probe under the published MERL protocol.

The MERL CPSC signal protocol is preserved (first 2,500 samples, zero pad to
5,000, record-wise min/max normalization, and aVL/aVF swap).  The only model
adapter is the deterministic conversion required by the released ST-MEM
ViT-B/75 encoder: 500 Hz -> 250 Hz and one center crop to 2,250 samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import signal
from scipy.io import loadmat
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, TensorDataset

from common import atomic_json, array_sha256, save_npy_atomic, sha256, state_dict
from protocol import STMEM_SHA256

CAMPAIGN = "q1-stmem-cpsc-merl-protocol-seed42-20260916"
SEED = 42
SPLITS = ("train", "val", "test")


def digest_strings(values: np.ndarray) -> str:
    return hashlib.sha256("\n".join(map(str, values.tolist())).encode()).hexdigest()


def load_waveform(path: Path) -> np.ndarray:
    payload = loadmat(path)
    arrays = [value for key, value in payload.items()
              if not key.startswith("__") and isinstance(value, np.ndarray) and value.ndim == 2]
    candidates = [value for value in arrays if 12 in value.shape]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one 12-lead matrix in {path}, found {len(candidates)}")
    value = np.asarray(candidates[0], dtype=np.float32)
    if value.shape[0] != 12:
        value = value.T
    return value


def merl_stmem_adapter(value: np.ndarray) -> np.ndarray:
    """Apply MERL preprocessing, then the minimum ST-MEM length adapter."""
    value = value[:, :2500]
    if value.shape[1] < 2500:
        value = np.pad(value, ((0, 0), (0, 2500 - value.shape[1])))
    value = np.pad(value, ((0, 0), (0, 2500)))
    lo, hi = float(value.min()), float(value.max())
    value = (value - lo) / (hi - lo + 1e-8)
    value[[4, 5]] = value[[5, 4]]
    value = signal.resample(value, 2500, axis=-1).astype(np.float32)
    return np.ascontiguousarray(value[:, 125:2375])


class CPSC(Dataset):
    def __init__(self, array_root: Path, raw_root: Path, split: str):
        self.paths = np.load(array_root / f"{split}_path.npy", allow_pickle=True)
        self.labels = np.load(array_root / f"{split}_labels.npy", mmap_mode="r")
        self.raw_root = raw_root

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        name = str(self.paths[index])
        waveform = merl_stmem_adapter(load_waveform(self.raw_root / f"{name}.mat"))
        return torch.from_numpy(waveform), torch.from_numpy(
            np.array(self.labels[index], copy=True, dtype=np.float32)
        )


def build_encoder(root: Path):
    repository = root / "external_models/ST-MEM"
    sys.path.insert(0, str(repository))
    from models.encoder.st_mem_vit import st_mem_vit_base

    checkpoint = root / "model_weights/ST-MEM/st_mem_vit_base_encoder.pth"
    if sha256(checkpoint) != STMEM_SHA256:
        raise RuntimeError("official ST-MEM encoder SHA256 mismatch")
    model = st_mem_vit_base(num_leads=12, num_classes=None, seq_len=2250, patch_size=75)
    payload = {key.removeprefix("module."): value for key, value in
               state_dict(torch.load(checkpoint, map_location="cpu")).items()
               if not key.startswith("head.")}
    incompatible = model.load_state_dict(payload, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"encoder checkpoint mismatch: {incompatible}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model.cuda().eval(), checkpoint


@torch.inference_mode()
def extract(model, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for waveform, truth in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            encoded = model.forward_encoding(waveform.cuda(non_blocking=True).float())
        features.append(encoded.float().cpu().numpy())
        labels.append(truth.numpy())
    return np.concatenate(features).astype(np.float32), np.concatenate(labels).astype(np.float32)


@torch.inference_mode()
def predict(head: torch.nn.Module, features: torch.Tensor) -> np.ndarray:
    rows = []
    for start in range(0, len(features), 1024):
        rows.append(head(features[start:start + 1024].cuda()).float().cpu().numpy())
    return np.concatenate(rows)


def scores(truth: np.ndarray, logits: np.ndarray) -> dict:
    probability = torch.sigmoid(torch.from_numpy(logits.astype(np.float64))).numpy()
    return {
        "macro_auroc": float(roc_auc_score(truth, probability, average="macro")),
        "macro_auprc": float(average_precision_score(truth, probability, average="macro")),
        "per_class_auroc": [float(roc_auc_score(truth[:, i], probability[:, i])) for i in range(9)],
        "per_class_auprc": [float(average_precision_score(truth[:, i], probability[:, i])) for i in range(9)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    output = root / "results" / CAMPAIGN
    status = output / "queue-status.json"
    arrays = root / "src/CLEAR-HUG/datasets/ecg_datasets/CPSC2018/data"
    raw = root / "src/CLEAR-HUG/datasets/dataset_preprocess/CPSC2018/wfdb"
    output.mkdir(parents=True, exist_ok=True)

    random.seed(0); np.random.seed(0); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    model, encoder_checkpoint = build_encoder(root)
    split_manifest = {}
    feature_root = output / "features"

    try:
        for split in SPLITS[:2]:
            atomic_json(status, {"state": "running", "stage": f"extract-{split}"})
            dataset = CPSC(arrays, raw, split)
            features, labels = extract(model, DataLoader(
                dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True
            ))
            save_npy_atomic(feature_root / f"{split}_features.npy", features)
            save_npy_atomic(feature_root / f"{split}_labels.npy", labels)
            split_manifest[split] = {
                "records": len(labels), "features_sha256": array_sha256(features),
                "labels_sha256": array_sha256(labels),
                "record_ids_sha256": digest_strings(dataset.paths),
            }

        atomic_json(status, {"state": "running", "stage": "linear-probe"})
        train_x = torch.from_numpy(np.load(feature_root / "train_features.npy"))
        train_y = torch.from_numpy(np.load(feature_root / "train_labels.npy"))
        val_x = torch.from_numpy(np.load(feature_root / "val_features.npy"))
        val_y = np.load(feature_root / "val_labels.npy")
        generator = torch.Generator().manual_seed(SEED)
        loader = DataLoader(TensorDataset(train_x, train_y), batch_size=16, shuffle=True,
                            generator=generator, num_workers=0)
        head = torch.nn.Linear(768, 9).cuda()
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.1)
        criterion = torch.nn.BCEWithLogitsLoss()
        best = None; log = []
        for epoch in range(100):
            head.train(); losses = []
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(head(batch_x.cuda()), batch_y.cuda())
                loss.backward(); optimizer.step()
                # Preserve the released MERL source behavior exactly.
                scheduler.step(); losses.append(float(loss.detach().cpu()))
            scheduler.step()
            head.eval(); logits = predict(head, val_x); metric = scores(val_y, logits)
            row = {"epoch": epoch + 1, "loss": float(np.mean(losses)),
                   "learning_rate": optimizer.param_groups[0]["lr"], **metric}
            log.append(row)
            print(json.dumps(row), flush=True)
            if best is None or metric["macro_auroc"] > best["macro_auroc"]:
                best = {"epoch": epoch + 1, **metric}
                torch.save({"model": head.state_dict(), "epoch": epoch + 1,
                            "validation": metric}, output / "checkpoint-best.pth")
        atomic_json(output / "training-log.json", {"epochs": log})

        atomic_json(status, {"state": "running", "stage": "formal-test"})
        dataset = CPSC(arrays, raw, "test")
        test_x, test_y = extract(model, DataLoader(
            dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True
        ))
        split_manifest["test"] = {
            "records": len(test_y), "features_sha256": array_sha256(test_x),
            "labels_sha256": array_sha256(test_y),
            "record_ids_sha256": digest_strings(dataset.paths),
        }
        payload = torch.load(output / "checkpoint-best.pth", map_location="cpu")
        head.load_state_dict(payload["model"]); head.eval()
        logits = predict(head, torch.from_numpy(test_x))
        probability = torch.sigmoid(torch.from_numpy(logits.astype(np.float64))).numpy()
        result = {"status": "complete", "campaign": CAMPAIGN, "seed": SEED,
                  "best_validation": best, "test": scores(test_y, logits),
                  "checkpoint_sha256": sha256(output / "checkpoint-best.pth")}
        formal = output / "formal-test"; formal.mkdir(parents=True, exist_ok=True)
        incoming = formal / "test_predictions.npz.incoming"
        with incoming.open("wb") as handle:
            np.savez_compressed(handle, y_true=test_y, logits=logits,
                                probabilities=probability, record_ids=dataset.paths)
        os.replace(incoming, formal / "test_predictions.npz")
        atomic_json(formal / "result.json", result)

        manifest = {
            "status": "complete", "campaign": CAMPAIGN, "seed": SEED,
            "encoder": "official ST-MEM ViT-B/75 pretrained encoder",
            "encoder_sha256": sha256(encoder_checkpoint), "encoder_frozen": True,
            "trainable_parameters": sum(p.numel() for p in head.parameters()),
            "optimizer": "Adam", "learning_rate": 1e-3, "weight_decay": 1e-4,
            "batch_size": 16, "epochs": 100,
            "scheduler": "MultiStepLR(milestone=40,gamma=0.1); released MERL source steps per batch and epoch",
            "selection_metric": "validation_macro_auroc", "test_used_for_selection": False,
            "cpsc_protocol": "MERL: first 2500 @500Hz, pad to 5000, global minmax [0,1], swap aVL/aVF",
            "stmem_length_adapter": "resample 500Hz->250Hz, one center crop 2500->2250; no multi-crop",
            "stmem_extra_filters_or_standardization": False,
            "split_counts": {key: value["records"] for key, value in split_manifest.items()},
            "splits": split_manifest,
        }
        atomic_json(output / "manifest.json", manifest)
        atomic_json(status, {"state": "complete", "stage": "complete"})
        print(json.dumps(result, indent=2), flush=True)
    except Exception as exc:
        atomic_json(status, {"state": "failed", "stage": "unknown", "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
