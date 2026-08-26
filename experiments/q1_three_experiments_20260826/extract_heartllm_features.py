"""Extract frozen HeartLLM encoder lead features for PTB-XL train/val."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb

from common import ensure_development_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ptbxl", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    ensure_development_paths(args.ptbxl, args.records, args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for HeartLLM encoder extraction")
    sys.path.insert(0, str(args.root / "src/HeartLLM/ecg_tokenizer"))
    from tokenizer import ECGEncoder
    sys.path.insert(0, str(args.root / "mvp"))
    from train_fsq_pilot import build_labels

    labels, classes = build_labels(args.ptbxl, "superdiag")
    metadata = pd.read_csv(args.ptbxl / "ptbxl_database.csv")
    metadata = metadata.loc[metadata["strat_fold"].isin(list(range(1, 10)))].copy()
    if metadata["strat_fold"].eq(10).any():
        raise RuntimeError("test fold unexpectedly present")
    encoder = ECGEncoder().cuda().eval()
    payload = torch.load(args.checkpoint, map_location="cpu")
    state = payload.get("model", payload)
    enc_state = {key[len("enc."):]: value for key, value in state.items() if key.startswith("enc.")}
    incompatible = encoder.load_state_dict(enc_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"HeartLLM encoder checkpoint mismatch: {incompatible}")
    by_split = {
        "train": metadata.loc[metadata["strat_fold"] <= 8].reset_index(drop=True),
        "val": metadata.loc[metadata["strat_fold"] == 9].reset_index(drop=True),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "scope": "development train/validation only",
        "formal_test_used": False,
        "encoder": "HeartLLM ECGEncoder (frozen)",
        "checkpoint": str(args.checkpoint),
        "classes": classes,
        "splits": {},
    }
    started = time.perf_counter()
    with torch.inference_mode():
        for split, rows in by_split.items():
            destination = args.output / split
            destination.mkdir(exist_ok=True)
            total = len(rows)
            features = np.lib.format.open_memmap(destination / "features.npy", mode="w+", dtype=np.float16, shape=(total, 12, 32))
            ecg_ids = np.lib.format.open_memmap(destination / "ecg_id.npy", mode="w+", dtype=np.int32, shape=(total,))
            folds = np.lib.format.open_memmap(destination / "fold.npy", mode="w+", dtype=np.uint8, shape=(total,))
            targets = np.lib.format.open_memmap(destination / "labels.npy", mode="w+", dtype=np.float32, shape=(total, len(classes)))
            pending, pending_ids, pending_folds, pending_targets = [], [], [], []
            offset = 0
            failures = []

            def flush() -> None:
                nonlocal offset
                if not pending:
                    return
                signals = torch.from_numpy(np.stack(pending)).cuda(non_blocking=True).float()
                with torch.autocast("cuda", dtype=torch.float16):
                    per_lead = []
                    for lead in range(12):
                        per_lead.append(encoder._enc(signals[:, lead:lead + 1]).mean(dim=-1))
                    encoded = torch.stack(per_lead, dim=1)
                end = offset + len(pending)
                features[offset:end] = encoded.float().cpu().numpy().astype(np.float16)
                ecg_ids[offset:end] = np.asarray(pending_ids, dtype=np.int32)
                folds[offset:end] = np.asarray(pending_folds, dtype=np.uint8)
                targets[offset:end] = np.asarray(pending_targets, dtype=np.float32)
                offset = end
                pending.clear(); pending_ids.clear(); pending_folds.clear(); pending_targets.clear()

            for _, row in rows.iterrows():
                ecg_id = int(row["ecg_id"])
                try:
                    path = args.records / str(row["filename_hr"])
                    ecg, fields = wfdb.rdsamp(str(path))
                    if int(fields["fs"]) != 500 or ecg.shape[0] != 5000 or ecg.shape[1] != 12:
                        raise ValueError(f"unexpected record shape/fs: {ecg.shape}, {fields.get('fs')}")
                    ecg = np.asarray(ecg, dtype=np.float32).T
                    ecg = (ecg - ecg.mean(axis=1, keepdims=True)) / np.maximum(ecg.std(axis=1, keepdims=True), 1.0e-6)
                    pending.append(ecg)
                    pending_ids.append(ecg_id)
                    pending_folds.append(int(row["strat_fold"]))
                    pending_targets.append(labels[ecg_id])
                    if len(pending) >= args.batch_size:
                        flush()
                except Exception as exc:
                    failures.append({"ecg_id": ecg_id, "error": repr(exc)})
            flush()
            for array in (features, ecg_ids, folds, targets):
                array.flush()
            if offset != total - len(failures):
                raise RuntimeError(f"{split}: wrote {offset}, expected {total - len(failures)}")
            if failures:
                raise RuntimeError(f"{split}: {len(failures)} records could not be read; refusing an incomplete cache")
            (destination / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
            manifest["splits"][split] = {"records": int(offset), "requested": total, "failures": len(failures), "path": str(destination)}
    manifest["elapsed_seconds"] = time.perf_counter() - started
    incoming = args.output / "feature-manifest.json.incoming"
    incoming.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(incoming, args.output / "feature-manifest.json")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
