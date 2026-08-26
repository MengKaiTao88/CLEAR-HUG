"""Extract frozen CLEAR local features for train/val only.

This is intentionally independent of the older cache helper: several server
copies have a validation ``*_data_in_chans.npy`` file missing even though the
full 12-lead QRS array is present.  For a full lead set the channel IDs are
deterministic (1..12, fifteen beats each), so they can be reconstructed without
touching a test split.
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
        self.split = split
        self.root = root
        self.data = np.load(root / f"{split}_data.npy", mmap_mode="r")
        labels_root = labels_root or root
        label_path = labels_root / f"{split}_labels.npy"
        if not label_path.is_file():
            raise FileNotFoundError(
                f"{split}: labels not found at {label_path}; pass --labels-root "
                "when labels are stored outside the QRS token directory"
            )
        self.labels = np.load(label_path, mmap_mode="r")
        self.times = np.load(root / f"{split}_data_in_times.npy", mmap_mode="r")
        channel_path = root / f"{split}_data_in_chans.npy"
        if channel_path.is_file():
            self.channels = np.load(channel_path, mmap_mode="r")
        else:
            # CLEAR uses 1-based spatial IDs; each lead contributes 15 tokens.
            self.channels = np.tile(np.repeat(np.arange(1, 13, dtype=np.int64), 15), (len(self.data), 1))
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=None,
        help="Optional directory containing train/val_labels.npy (e.g. CPSC2018/data).",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional trained CLEAR-DeepSets checkpoint. The released checkpoint "
            "only contains the frozen backbone; when this argument is supplied "
            "its adapter/head are used to write meaningful baseline logits."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=6)
    args = parser.parse_args()

    ensure_development_paths(args.dataset, args.output, args.labels_root or args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(args.root / "src/CLEAR-HUG"), str(args.root / "mvp")]
    import modeling_finetune_fair_baselines  # noqa: F401
    from clear_adapter import ClearFeatureAdapter

    baseline_available = args.baseline_checkpoint is not None
    baseline_model = None
    if baseline_available:
        baseline_model = create_model(
            "CLEAR_MASKED_DEEPSETS_finetune_base",
            pretrained=False,
            num_classes=args.classes,
            cls_token_num=12,
            padding_mask=False,
            atten_mask=True,
            mask_ratio=0,
        )
        baseline_payload = torch.load(args.baseline_checkpoint, map_location="cpu")
        baseline_state = baseline_payload.get("model", baseline_payload)
        incompatible = baseline_model.load_state_dict(baseline_state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"baseline checkpoint mismatch: {incompatible}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen CLEAR feature extraction")
    # Use the deterministic feature-only path.  The official forward_feature
    # shuffles local tokens even when mask_ratio=0, which makes Lead-CLS values
    # (and therefore cached baseline logits) depend on a random permutation.
    feature_model = ClearFeatureAdapter(
        args.root / "src/CLEAR-HUG", args.checkpoint
    ).cuda().eval()
    if baseline_model is not None:
        baseline_model.cuda().eval()

    manifest = {
        "schema_version": 1,
        "scope": "development train/validation only",
        "formal_test_used": False,
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "baseline_checkpoint": (
            str(args.baseline_checkpoint) if args.baseline_checkpoint else None
        ),
        "baseline_available": baseline_available,
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
                    lead_valid = valid.any(dim=1)
                    if baseline_available:
                        record_shared = baseline_model.adapter(lead_cls, lead_valid)
                        baseline_logits = baseline_model.mlp_head(record_shared)
                        flat_local = local.reshape(batch * 15, 12, 768)
                        flat_valid = valid.reshape(batch * 15, 12)
                        beat_shared = baseline_model.adapter(flat_local, flat_valid).reshape(
                            batch, 15, 768
                        )
                    else:
                        # Attention-only runs do not consume baseline logits, but
                        # keep the array finite and make its provenance explicit
                        # in the manifest rather than silently using random heads.
                        beat_shared = torch.zeros(
                            (batch, 15, 768),
                            device=lead_cls.device,
                            dtype=lead_cls.dtype,
                        )
                        baseline_logits = torch.zeros(
                            (batch, args.classes),
                            device=lead_cls.device,
                            dtype=lead_cls.dtype,
                        )
                end = offset + len(signals)
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
    temporary = args.output / "feature-manifest.json.incoming"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, args.output / "feature-manifest.json")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
