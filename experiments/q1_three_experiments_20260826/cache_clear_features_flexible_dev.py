"""Cache frozen CLEAR features for development-only ablation runs.

This is the same feature-only path as ``cache_clear_features.py`` but lets a
caller select the matching frozen baseline architecture (for example HUG for
the retained PTB-XL Sub checkpoint, or masked DeepSets for Form/CSN).  Only
train and val arrays are opened.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from timm.models import create_model

from common import ensure_development_paths


class QRSDevelopmentDataset(Dataset):
    def __init__(self, root: Path, split: str, labels_root: Path | None = None):
        if split not in {"train", "val"}:
            raise ValueError(split)
        ensure_development_paths(root)
        self.root = root
        self.split = split
        self.data = np.load(root / f"{split}_data.npy", mmap_mode="r")
        labels_root = labels_root or root
        self.labels = np.load(labels_root / f"{split}_labels.npy", mmap_mode="r")
        self.times = np.load(root / f"{split}_data_in_times.npy", mmap_mode="r")
        channel_path = root / f"{split}_data_in_chans.npy"
        if channel_path.is_file():
            self.channels = np.load(channel_path, mmap_mode="r")
        else:
            self.channels = np.tile(
                np.repeat(np.arange(1, 13, dtype=np.int64), 15),
                (len(self.data), 1),
            )
        if len(self.data) != len(self.labels) or len(self.data) != len(self.times):
            raise RuntimeError(f"{root}/{split}: QRS arrays have inconsistent lengths")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return (
            np.asarray(self.data[index], dtype=np.float32),
            np.asarray(self.labels[index], dtype=np.float32),
            np.asarray(self.channels[index], dtype=np.int64),
            np.asarray(self.times[index], dtype=np.int64),
        )


def open_memmap(path: Path, shape: tuple[int, ...], dtype) -> np.memmap:
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def aggregate_baseline(model, lead_cls, local, valid):
    """Return record logits, beat summaries and the lead-valid mask."""

    # The fair-baseline HUG and DeepSets adapters both expose the same masked
    # aggregation API; the only difference is which adapter is selected.
    from modeling_finetune_fair_baselines import masked_mean

    batch = local.shape[0]
    lead_valid = valid.any(dim=1)
    # Legacy CLEAR_HUG checkpoints use the original stage-2 classifier, whose
    # record head is a plain mean over the twelve Lead-CLS tokens and has no
    # named aggregation adapter.
    if hasattr(model, "stage") and not hasattr(model, "aggregation"):
        summary = lead_cls.mean(dim=1)
        baseline_logits = model.mlp_head(summary)
        beat_shared = local.mean(dim=2)
        return baseline_logits, beat_shared, lead_valid

    aggregation = getattr(model, "aggregation", "masked_deepsets")
    if aggregation in {"masked_hug", "hug_random_lead_dropout"}:
        groups, group_valid = model.adapter(lead_cls, lead_valid)
        summary, _ = masked_mean(groups, group_valid, dim=1)
    elif aggregation in {"masked_deepsets", "masked_deepsets_random_lead_dropout"}:
        summary = model.adapter(lead_cls, lead_valid)
    elif aggregation == "masked_mean":
        summary, _ = masked_mean(lead_cls, lead_valid, dim=1)
    else:
        raise RuntimeError(f"unsupported baseline aggregation {aggregation!r}")
    baseline_logits = model.mlp_head(summary)
    flat_local = local.reshape(batch * 15, 12, 768)
    flat_valid = valid.reshape(batch * 15, 12)
    if aggregation in {"masked_hug", "hug_random_lead_dropout"}:
        flat_lead_valid = flat_valid
        flat_groups, flat_group_valid = model.adapter(flat_local, flat_lead_valid)
        beat_shared, _ = masked_mean(flat_groups, flat_group_valid, dim=1)
    elif aggregation in {"masked_deepsets", "masked_deepsets_random_lead_dropout"}:
        beat_shared = model.adapter(flat_local, flat_valid)
    else:
        beat_shared, _ = masked_mean(flat_local, flat_valid, dim=1)
    return baseline_logits, beat_shared.reshape(batch, 15, 768), lead_valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-model", default="CLEAR_MASKED_DEEPSETS_finetune_base")
    parser.add_argument("--labels-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=6)
    args = parser.parse_args()
    ensure_development_paths(
        args.dataset,
        args.output,
        args.baseline_checkpoint,
        args.labels_root or args.dataset,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(args.root / "src/CLEAR-HUG"), str(args.root / "mvp")]
    import modeling_finetune_fair_baselines  # noqa: F401
    import modeling_finetune  # noqa: F401  (registers legacy CLEAR_HUG)
    from clear_adapter import ClearFeatureAdapter

    baseline_model = create_model(
        args.baseline_model,
        pretrained=False,
        num_classes=args.classes,
        cls_token_num=12,
        padding_mask=False,
        atten_mask=True,
        mask_ratio=0,
    )
    checkpoint = torch.load(args.baseline_checkpoint, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    incompatible = baseline_model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint/model mismatch: missing={incompatible.missing_keys[:8]} "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen CLEAR feature extraction")
    feature_model = ClearFeatureAdapter(
        args.root / "src/CLEAR-HUG", args.checkpoint
    ).cuda().eval()
    baseline_model.cuda().eval()

    manifest = {
        "schema_version": 1,
        "scope": "development train/validation only",
        "formal_test_used": False,
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "baseline_model": args.baseline_model,
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "classes": args.classes,
        "splits": {},
    }
    with torch.inference_mode():
        for split in ("train", "val"):
            destination = args.output / split
            destination.mkdir(exist_ok=True)
            dataset = QRSDevelopmentDataset(args.dataset, split, args.labels_root)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            total = len(dataset)
            local_out = open_memmap(destination / "local.npy", (total, 15, 12, 768), np.float16)
            shared_out = open_memmap(destination / "shared.npy", (total, 15, 768), np.float16)
            valid_out = open_memmap(destination / "valid.npy", (total, 15, 12), np.bool_)
            logits_out = open_memmap(destination / "baseline_logits.npy", (total, args.classes), np.float32)
            labels_out = open_memmap(destination / "labels.npy", (total, args.classes), np.float32)
            offset = 0
            for signals, labels, channels, times in loader:
                signals = signals.cuda(non_blocking=True).float()
                channels = channels.cuda(non_blocking=True)
                times = times.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    local, lead_cls = feature_model(signals, channels, times)
                    batch = len(signals)
                    valid = times.reshape(batch, 12, 15).gt(0).permute(0, 2, 1)
                    baseline_logits, beat_shared, _ = aggregate_baseline(
                        baseline_model, lead_cls, local, valid
                    )
                end = offset + batch
                local_out[offset:end] = local.float().cpu().numpy().astype(np.float16)
                shared_out[offset:end] = beat_shared.float().cpu().numpy().astype(np.float16)
                valid_out[offset:end] = valid.cpu().numpy()
                logits_out[offset:end] = baseline_logits.float().cpu().numpy()
                labels_out[offset:end] = labels.numpy().astype(np.float32)
                offset = end
            for array in (local_out, shared_out, valid_out, logits_out, labels_out):
                array.flush()
            if offset != total:
                raise RuntimeError(f"{split}: wrote {offset}, expected {total}")
            manifest["splits"][split] = {"records": total, "path": str(destination)}

    incoming = args.output / "feature-manifest.json.incoming"
    incoming.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(incoming, args.output / "feature-manifest.json")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
