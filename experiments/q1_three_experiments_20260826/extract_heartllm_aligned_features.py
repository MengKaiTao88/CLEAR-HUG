"""Extract frozen HeartLLM features for heartbeat-by-lead QRS tokens.

The earlier HeartLLM extraction averaged every lead over the whole ten-second
record and therefore produced only one token per lead.  This development-only
extractor keeps the QRS grid (15 heartbeats x 12 leads) and applies the frozen
HeartLLM convolutional encoder independently to each 96-sample local segment.
No formal-test arrays are opened.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import ensure_development_paths


def open_memmap(path: Path, shape: tuple[int, ...], dtype) -> np.memmap:
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def to_grid(signals: torch.Tensor, channels: torch.Tensor, times: torch.Tensor):
    """Map lead-major QRS tokens to (batch, 15 beats, 12 leads, 96 samples)."""

    batch, tokens, width = signals.shape
    if tokens != 180 or width != 96:
        raise RuntimeError(f"expected QRS shape (B,180,96), got {tuple(signals.shape)}")
    lead = channels.to(torch.long) - 1
    beat = times.to(torch.long) - 1
    valid = (lead >= 0) & (lead < 12) & (beat >= 0) & (beat < 15)
    flat = (beat.clamp(0, 14) * 12 + lead.clamp(0, 11)).to(torch.long)
    grid = torch.zeros((batch, 180, width), dtype=signals.dtype, device=signals.device)
    safe_signals = signals * valid.unsqueeze(-1).to(signals.dtype)
    grid.scatter_(1, flat.unsqueeze(-1).expand(-1, -1, width), safe_signals)
    grid = grid.reshape(batch, 15, 12, width)
    valid_grid = torch.zeros((batch, 180), dtype=torch.bool, device=signals.device)
    valid_grid.scatter_(1, flat, valid)
    return grid, valid_grid.reshape(batch, 15, 12)


def encode_segments(encoder, grid: torch.Tensor, segment_batch: int) -> torch.Tensor:
    batch = grid.shape[0]
    flat = grid.reshape(batch * 180, 1, 96)
    outputs = []
    for start in range(0, len(flat), segment_batch):
        chunk = flat[start : start + segment_batch]
        # HeartLLM was trained on per-channel standardized ECG.  QRS windows
        # retain that local semantics, so normalize each heartbeat/lead token
        # independently before the frozen encoder.
        mean = chunk.mean(dim=-1, keepdim=True)
        std = chunk.std(dim=-1, keepdim=True).clamp_min(1.0e-6)
        chunk = (chunk - mean) / std
        with torch.autocast("cuda", dtype=torch.float16):
            outputs.append(encoder._enc(chunk).mean(dim=-1))
    return torch.cat(outputs, dim=0).reshape(batch, 15, 12, 32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--segment-batch", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    del args.num_workers  # QRS memmaps are read directly; no worker copies are needed.
    ensure_development_paths(args.dataset, args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for HeartLLM feature extraction")

    sys.path.insert(0, str(args.root / "src/HeartLLM/ecg_tokenizer"))
    from tokenizer import ECGEncoder

    encoder = ECGEncoder().cuda().eval()
    payload = torch.load(args.checkpoint, map_location="cpu")
    state = payload.get("model", payload)
    enc_state = {key[len("enc.") :]: value for key, value in state.items() if key.startswith("enc.")}
    if not enc_state:
        raise RuntimeError("HeartLLM checkpoint does not contain enc.* weights")
    incompatible = encoder.load_state_dict(enc_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"HeartLLM encoder checkpoint mismatch: {incompatible}")

    manifest = {
        "schema_version": 1,
        "scope": "development train/validation only",
        "formal_test_used": False,
        "encoder": "HeartLLM ECGEncoder (frozen)",
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "classes": args.classes,
        "feature_shape": ["N", 15, 12, 32],
        "token_layout": "heartbeat x lead x HeartLLM local embedding",
        "source_window_samples": 96,
        "normalization": "per heartbeat-lead window z-score",
        "splits": {},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    with torch.inference_mode():
        for split in ("train", "val"):
            data = np.load(args.dataset / f"{split}_data.npy", mmap_mode="r")
            labels = np.load(args.dataset / f"{split}_labels.npy", mmap_mode="r")
            channels_path = args.dataset / f"{split}_data_in_chans.npy"
            times_path = args.dataset / f"{split}_data_in_times.npy"
            channels = np.load(channels_path, mmap_mode="r") if channels_path.is_file() else np.tile(
                np.repeat(np.arange(1, 13, dtype=np.int64), 15), (len(data), 1)
            )
            times = np.load(times_path, mmap_mode="r")
            if data.ndim != 3 or data.shape[1:] != (180, 96):
                raise RuntimeError(f"{split}: expected (N,180,96), got {data.shape}")
            if labels.ndim != 2 or labels.shape[1] != args.classes or len(data) != len(labels):
                raise RuntimeError(f"{split}: labels/data shape mismatch")
            destination = args.output / split
            destination.mkdir(exist_ok=True)
            feature_out = open_memmap(destination / "features.npy", (len(data), 15, 12, 32), np.float16)
            valid_out = open_memmap(destination / "valid.npy", (len(data), 15, 12), np.bool_)
            labels_out = open_memmap(destination / "labels.npy", (len(data), args.classes), np.float32)
            offset = 0
            for start in range(0, len(data), args.batch_size):
                end = min(start + args.batch_size, len(data))
                signals = torch.from_numpy(np.asarray(data[start:end])).cuda(non_blocking=True).float()
                batch_channels = torch.from_numpy(np.asarray(channels[start:end])).cuda(non_blocking=True)
                batch_times = torch.from_numpy(np.asarray(times[start:end])).cuda(non_blocking=True)
                grid, valid = to_grid(signals, batch_channels, batch_times)
                encoded = encode_segments(encoder, grid, args.segment_batch)
                feature_out[start:end] = encoded.float().cpu().numpy().astype(np.float16)
                valid_out[start:end] = valid.cpu().numpy()
                labels_out[start:end] = np.asarray(labels[start:end], dtype=np.float32)
                offset = end
                if offset % max(args.batch_size * 20, args.batch_size) == 0:
                    print(json.dumps({"split": split, "records": offset, "total": len(data)}), flush=True)
            for array in (feature_out, valid_out, labels_out):
                array.flush()
            if offset != len(data):
                raise RuntimeError(f"{split}: wrote {offset}, expected {len(data)}")
            manifest["splits"][split] = {"records": int(len(data)), "path": str(destination)}

    manifest["elapsed_seconds"] = time.perf_counter() - started
    incoming = args.output / "feature-manifest.json.incoming"
    incoming.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(incoming, args.output / "feature-manifest.json")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
