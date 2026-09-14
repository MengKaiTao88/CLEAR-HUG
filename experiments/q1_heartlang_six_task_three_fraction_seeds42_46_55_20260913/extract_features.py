#!/usr/bin/env python3
"""Extract deterministic frozen HeartLang or ST-MEM record features."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import atomic_json, save_npy_atomic, sha256, state_dict
from protocol import (
    CAMPAIGN,
    HEARTLANG_SHA256,
    STMEM_SHA256,
    TASKS,
)


class HeartLangDataset(Dataset):
    def __init__(self, root: Path, split: str):
        self.x = np.load(root / f"{split}_data.npy", mmap_mode="r")
        self.y = np.load(root / f"{split}_labels.npy", mmap_mode="r")
        self.channels = np.load(root / f"{split}_data_in_chans.npy", mmap_mode="r")
        self.times = np.load(root / f"{split}_data_in_times.npy", mmap_mode="r")
        if self.x.shape[1:] != (256, 96):
            raise RuntimeError(f"unexpected HeartLang input shape {self.x.shape}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return tuple(np.asarray(value[index]) for value in (self.x, self.y, self.channels, self.times))


class STMEMDataset(Dataset):
    def __init__(self, root: Path, split: str, transform):
        self.x = np.load(root / f"{split}_data.npy", mmap_mode="r")
        self.y = np.load(root / f"{split}_labels.npy", mmap_mode="r")
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        value = np.asarray(self.x[index], dtype=np.float32)
        if value.shape == (1000, 12):
            value = value.T
        if value.shape != (12, 1000):
            raise RuntimeError(f"unexpected ST-MEM raw shape {value.shape}")
        return self.transform(value), np.asarray(self.y[index], dtype=np.float32)


def load_heartlang(root: Path, checkpoint: Path):
    repo = root / "external_models/HeartLang"
    sys.path.insert(0, str(repo))
    import modeling_pretrain

    # The released pretraining checkpoint is an ST_ECGFormer state dict with
    # unprefixed backbone keys plus the language-model head.  Load that exact
    # architecture strictly and read its CLS token; wrapping it in the
    # fine-tuning classifier would introduce a spurious ``backbone.`` prefix.
    model = modeling_pretrain.HeartLang(pretrained=False)
    payload = state_dict(torch.load(checkpoint, map_location="cpu"))
    payload = {key.removeprefix("module."): value for key, value in payload.items()}
    incompatible = model.load_state_dict(payload, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"HeartLang checkpoint mismatch: {incompatible}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, incompatible


def load_stmem(root: Path, checkpoint: Path):
    repo = root / "external_models/ST-MEM"
    sys.path.insert(0, str(repo))
    from models.encoder.st_mem_vit import st_mem_vit_base
    import util.transforms as transforms

    model = st_mem_vit_base(num_leads=12, num_classes=None, seq_len=2250, patch_size=75)
    payload = state_dict(torch.load(checkpoint, map_location="cpu"))
    payload = {key.removeprefix("module."): value for key, value in payload.items() if not key.startswith("head.")}
    incompatible = model.load_state_dict(payload, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"ST-MEM checkpoint mismatch: {incompatible}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    preprocessing = transforms.Compose([
        transforms.Resample(target_fs=250),
        transforms.NCrop(crop_length=2250, num_segments=3),
        transforms.HighpassFilter(fs=250, cutoff=0.67),
        transforms.LowpassFilter(fs=250, cutoff=40),
        transforms.Standardize(axis=(-1, -2)),
        transforms.ToTensor(),
    ])

    def transform(value):
        # The frozen benchmark sources are all 100 Hz, so the resampler receives
        # the source frequency explicitly before deterministic three-crop eval.
        result = preprocessing.transforms[0](value, 100)
        for operation in preprocessing.transforms[1:]:
            result = operation(result)
        return result

    return model, transform, incompatible


def record_ids(qrs: Path, split: str, size: int) -> np.ndarray:
    path = qrs / f"{split}_path.npy"
    return np.load(path, allow_pickle=True) if path.exists() else np.arange(size, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", choices=("heartlang", "stmem"), required=True)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        manifest = args.output / "feature-manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()).get("status") == "complete":
            return
        raise RuntimeError(f"refusing incomplete existing output {args.output}")
    if args.split == "test":
        if not args.gate or not args.gate.exists():
            raise RuntimeError("formal test features require the global gate")
        gate = json.loads(args.gate.read_text())
        if gate.get("status") != "passed" or gate.get("formal_test_authorized") is not True:
            raise RuntimeError("global gate did not authorize formal test")
    elif args.gate:
        raise RuntimeError("gate is only accepted for test extraction")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    classes, qrs_rel, raw_rel, _ = TASKS[args.task]
    reference_qrs = args.root / "campaign-inputs" / "q1-clear-deepsets-hilar-10seed-20260903" / "ecg_datasets-resolved-v2" / qrs_rel
    heartlang_qrs = args.root / "campaign-inputs" / CAMPAIGN / "heartlang-qrs" / qrs_rel
    raw = args.root / "campaign-inputs" / CAMPAIGN / "raw" / raw_rel
    reference_labels = np.load(reference_qrs / f"{args.split}_labels.npy", mmap_mode="r")
    ids = record_ids(reference_qrs, args.split, len(reference_labels))
    checkpoint = args.root / "model_weights" / ("HeartLang/checkpoint-200.pth" if args.model == "heartlang" else "ST-MEM/st_mem_vit_base_encoder.pth")
    expected_sha = HEARTLANG_SHA256 if args.model == "heartlang" else STMEM_SHA256
    if sha256(checkpoint) != expected_sha:
        raise RuntimeError("official checkpoint SHA256 mismatch")

    if args.model == "heartlang":
        dataset = HeartLangDataset(heartlang_qrs, args.split)
        if dataset.y.shape != reference_labels.shape or not np.array_equal(dataset.y, reference_labels):
            raise RuntimeError("official HeartLang QRS labels/order differ from benchmark reference")
        model, incompatible = load_heartlang(args.root, checkpoint)
        batch_size = args.batch_size or 64
        transform_desc = "official HeartLang QRS sentence (256x96), CLS token"
    else:
        raw_labels = np.load(raw / f"{args.split}_labels.npy", mmap_mode="r")
        if raw_labels.shape != reference_labels.shape or not np.array_equal(raw_labels, reference_labels):
            raise RuntimeError("raw/QRS labels or record order differ")
        model, transform, incompatible = load_stmem(args.root, checkpoint)
        dataset = STMEMDataset(raw, args.split, transform)
        batch_size = args.batch_size or 8
        transform_desc = "100Hz->250Hz, deterministic three-crop 2250, filtered, standardized, mean feature"
    if len(dataset) != len(reference_labels) or len(ids) != len(reference_labels):
        raise RuntimeError("data/label/record-id length mismatch")
    model.cuda().eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    features = []
    with torch.inference_mode():
        for batch in loader:
            if args.model == "heartlang":
                signals, _, channels, times = batch
                with torch.autocast("cuda", dtype=torch.float16):
                    output = model(signals.cuda().float(), in_chan_matrix=channels.cuda(), in_time_matrix=times.cuda(), return_all_tokens=True)[:, 0]
            else:
                signals, _ = batch
                batch_n, crops, leads, length = signals.shape
                with torch.autocast("cuda", dtype=torch.float16):
                    output = model.forward_encoding(signals.cuda().float().reshape(batch_n * crops, leads, length))
                    output = output.reshape(batch_n, crops, -1).mean(dim=1)
            features.append(output.float().cpu().numpy())
    values = np.concatenate(features).astype(np.float32, copy=False)
    labels = np.asarray(reference_labels, dtype=np.float32)
    if values.shape != (len(labels), 768):
        raise RuntimeError(f"unexpected feature shape {values.shape}")
    args.output.mkdir(parents=True)
    save_npy_atomic(args.output / "features.npy", values)
    save_npy_atomic(args.output / "labels.npy", labels)
    save_npy_atomic(args.output / "record_ids.npy", np.asarray(ids))
    payload = {
        "status": "complete", "campaign": CAMPAIGN, "model": args.model,
        "task": args.task, "split": args.split, "records": len(labels),
        "classes": classes, "feature_shape": list(values.shape),
        "encoder_checkpoint": str(checkpoint), "encoder_sha256": expected_sha,
        "encoder_frozen": all(not p.requires_grad for p in model.parameters()),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "preprocessing": transform_desc,
        "features_sha256": sha256(args.output / "features.npy"),
        "labels_sha256": sha256(args.output / "labels.npy"),
        "record_ids_sha256": sha256(args.output / "record_ids.npy"),
        "test_gate": str(args.gate) if args.gate else None,
    }
    atomic_json(args.output / "feature-manifest.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
