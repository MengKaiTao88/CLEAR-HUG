#!/usr/bin/env python3
"""Create the exact official 256-token HeartLang QRS representation."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from common import atomic_json, sha256
from protocol import CAMPAIGN, HEARTLANG_COMMIT, TASKS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args()
    if args.split == "test":
        if not args.gate or not args.gate.exists():
            raise RuntimeError("test QRS preprocessing requires the global gate")
        gate = json.loads(args.gate.read_text())
        if gate.get("status") != "passed" or gate.get("formal_test_authorized") is not True:
            raise RuntimeError("global gate did not authorize formal test")
    elif args.gate:
        raise RuntimeError("gate is only accepted for test preprocessing")

    _, qrs_rel, raw_rel, _ = TASKS[args.task]
    raw_root = args.root / "campaign-inputs" / CAMPAIGN / "raw" / raw_rel
    output = args.root / "campaign-inputs" / CAMPAIGN / "heartlang-qrs" / qrs_rel
    manifest_path = output / f"{args.split}-qrs-manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("status") == "complete":
        return
    output.mkdir(parents=True, exist_ok=True)
    final_paths = {
        "data": output / f"{args.split}_data.npy",
        "channels": output / f"{args.split}_data_in_chans.npy",
        "times": output / f"{args.split}_data_in_times.npy",
        "labels": output / f"{args.split}_labels.npy",
        "record_ids": output / f"{args.split}_path.npy",
    }
    if any(path.exists() for path in final_paths.values()):
        raise RuntimeError(f"refusing incomplete existing QRS output for {args.task}:{args.split}")

    repo = args.root / "external_models/HeartLang"
    sys.path.insert(0, str(repo))
    from QRSTokenizer import QRSTokenizer

    raw_path = raw_root / f"{args.split}_data.npy"
    labels_path = raw_root / f"{args.split}_labels.npy"
    ids_path = raw_root / f"{args.split}_path.npy"
    raw = np.load(raw_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    if len(raw) != len(labels):
        raise RuntimeError("raw signal/label length mismatch")
    tokenizer = QRSTokenizer(
        fs=100, max_len=256, token_len=96, save_path=str(output),
        stage=args.split, used_channels=list(range(12)),
    )
    incoming = {name: path.with_name(path.name + ".incoming") for name, path in final_paths.items()}
    data_out = np.lib.format.open_memmap(incoming["data"], mode="w+", dtype=np.float32, shape=(len(raw), 256, 96))
    channel_out = np.lib.format.open_memmap(incoming["channels"], mode="w+", dtype=np.int32, shape=(len(raw), 256))
    time_out = np.lib.format.open_memmap(incoming["times"], mode="w+", dtype=np.int32, shape=(len(raw), 256))
    for index in range(len(raw)):
        signal = np.asarray(raw[index], dtype=np.float32)
        if signal.shape == (1000, 12):
            signal = signal.T
        if signal.shape != (12, 1000):
            raise RuntimeError(f"unexpected raw shape {signal.shape}")
        qrs = tokenizer.qrs_detection(signal)
        segments = tokenizer.extract_qrs_segments(signal, qrs)
        sentence, channels, times = tokenizer.qrs_to_sequence(segments, qrs)
        data_out[index] = sentence
        channel_out[index] = channels
        time_out[index] = times
        if index % 500 == 0:
            print(f"{args.task}:{args.split} {index}/{len(raw)}", flush=True)
    del data_out, channel_out, time_out
    shutil.copyfile(labels_path, incoming["labels"])
    if ids_path.exists():
        shutil.copyfile(ids_path, incoming["record_ids"])
    else:
        with incoming["record_ids"].open("wb") as handle:
            np.save(handle, np.arange(len(raw), dtype=np.int64))
    for name, path in final_paths.items():
        os.replace(incoming[name], path)
    payload = {
        "status": "complete", "campaign": CAMPAIGN, "task": args.task,
        "split": args.split, "records": len(raw), "token_shape": [256, 96],
        "official_heartlang_commit": HEARTLANG_COMMIT,
        "tokenizer": {"fs": 100, "max_len": 256, "token_len": 96, "used_channels": list(range(12))},
        "source_raw_sha256": sha256(raw_path), "source_labels_sha256": sha256(labels_path),
        "data_sha256": sha256(final_paths["data"]), "labels_sha256": sha256(final_paths["labels"]),
        "record_ids_sha256": sha256(final_paths["record_ids"]),
        "test_gate": str(args.gate) if args.gate else None,
    }
    atomic_json(manifest_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
