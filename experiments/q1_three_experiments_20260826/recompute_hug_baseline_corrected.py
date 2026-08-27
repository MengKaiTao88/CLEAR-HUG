"""Recompute legacy HUG validation logits with the correct stage-2 MoE.

The original development cache used a plain mean over the twelve Lead-CLS
tokens for ``CLEAR_HUG_finetune_base``.  A stage-2 HUG checkpoint instead
applies ``backbone.moe`` and averages its seven group summaries before the
classification head.  This script recomputes only the train/val baseline
logits, reuses the already extracted local features through read-only
symlinks, and writes a new cache directory without touching the old cache or
any frozen result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from timm.models import create_model

from cache_clear_features import QRSDevelopmentDataset
from common import ensure_development_paths


def _state_dict(payload):
    if isinstance(payload, dict):
        for key in ("model", "state_dict"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                return candidate
        if payload and all(hasattr(value, "numel") for value in payload.values()):
            return payload
    raise RuntimeError("checkpoint has no tensor state dictionary")


def _metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float | int]:
    valid = [i for i in range(labels.shape[1]) if np.unique(labels[:, i]).size == 2]
    scores = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    return {
        "macro_auroc": float(roc_auc_score(labels[:, valid], scores[:, valid], average="macro")),
        "macro_auprc": float(
            average_precision_score(labels[:, valid], scores[:, valid], average="macro")
        ),
        "valid_classes": len(valid),
    }


def _autocast(device: torch.device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def _hug_logits(model, lead_cls: torch.Tensor) -> torch.Tensor:
    """Apply the exact stage-2 HUG record aggregation once."""

    if not hasattr(model, "backbone") or not hasattr(model.backbone, "moe"):
        raise RuntimeError("CLEAR_HUG checkpoint model has no backbone.moe")
    groups = model.backbone.moe(lead_cls)
    summary = torch.stack(groups, dim=1).mean(dim=1)
    return model.mlp_head(summary)


def _link_readonly(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing path: {target}")
    target.symlink_to(source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--qrs-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--released-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    paths = (
        args.qrs_root,
        args.labels_root,
        args.source_cache,
        args.output_cache,
        args.released_checkpoint,
        args.baseline_checkpoint,
    )
    for path in paths:
        ensure_development_paths(path)
        if "test" in {part.lower() for part in path.parts}:
            raise RuntimeError(f"development-only recomputation refuses test path: {path}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but CUDA is unavailable")

    args.output_cache.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(args.root / "src/CLEAR-HUG"), str(args.root / "mvp")]
    import modeling_finetune  # noqa: F401 (registers CLEAR_HUG)
    from clear_adapter import ClearFeatureAdapter

    model = create_model(
        "CLEAR_HUG_finetune_base",
        pretrained=False,
        num_classes=args.classes,
        cls_token_num=12,
        padding_mask=False,
        atten_mask=True,
        mask_ratio=0,
    )
    payload = torch.load(args.baseline_checkpoint, map_location="cpu")
    incompatible = model.load_state_dict(_state_dict(payload), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"HUG checkpoint mismatch: missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )

    device = torch.device(args.device)
    feature_model = ClearFeatureAdapter(args.root / "src/CLEAR-HUG", args.released_checkpoint)
    model = model.to(device).eval()
    feature_model = feature_model.to(device).eval()
    manifest = {
        "schema_version": 1,
        "scope": "development train/validation only",
        "formal_test_used": False,
        "task": "PTBXL Subdiagnostic",
        "classes": args.classes,
        "encoder_checkpoint": str(args.released_checkpoint),
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "baseline_model": "CLEAR_HUG_finetune_base",
        "baseline_aggregation": "backbone.moe seven group summaries then mean, followed by mlp_head",
        "source_cache": str(args.source_cache),
        "reused_arrays": ["local.npy", "shared.npy", "valid.npy", "labels.npy"],
        "device": str(device),
        "splits": {},
    }

    started = time.perf_counter()
    with torch.inference_mode():
        for split in ("train", "val"):
            source_split = args.source_cache / split
            destination = args.output_cache / split
            destination.mkdir()
            for name in ("local.npy", "shared.npy", "valid.npy", "labels.npy"):
                source = source_split / name
                if not source.is_file():
                    raise FileNotFoundError(source)
                _link_readonly(source, destination / name)

            dataset = QRSDevelopmentDataset(args.qrs_root, split, args.labels_root)
            if dataset.labels.shape[1] != args.classes:
                raise RuntimeError(f"{split}: expected {args.classes} classes, got {dataset.labels.shape[1]}")
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            logits_path = destination / "baseline_logits.npy"
            logits_tmp = destination / "baseline_logits.npy.incoming"
            logits_out = np.lib.format.open_memmap(
                logits_tmp, mode="w+", dtype=np.float32, shape=(len(dataset), args.classes)
            )
            labels = []
            offset = 0
            for signals, batch_labels, channels, times in loader:
                signals = signals.to(device, non_blocking=True).float()
                channels = channels.to(device, non_blocking=True)
                times = times.to(device, non_blocking=True)
                with _autocast(device):
                    _local, lead_cls = feature_model(signals, channels, times)
                    batch_logits = _hug_logits(model, lead_cls)
                end = offset + len(signals)
                logits_out[offset:end] = batch_logits.float().cpu().numpy()
                labels.append(batch_labels.numpy())
                offset = end
            logits_out.flush()
            del logits_out
            os.replace(logits_tmp, logits_path)
            truth = np.concatenate(labels, axis=0)
            logits = np.load(logits_path, mmap_mode="r")
            manifest["splits"][split] = {
                "records": len(dataset),
                "path": str(destination),
                "metrics": _metrics(truth, np.asarray(logits)),
            }

    manifest["elapsed_seconds"] = time.perf_counter() - started
    incoming = args.output_cache / "feature-manifest.json.incoming"
    incoming.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(incoming, args.output_cache / "feature-manifest.json")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
