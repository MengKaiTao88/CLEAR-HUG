#!/usr/bin/env python3
"""Cache strictly paired HILA features for a fixed low-label training subset."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from timm.models import create_model
from torch.utils.data import DataLoader, SequentialSampler


def open_array(path: Path, shape: tuple[int, ...], dtype):
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--splits", nargs="+", default=("train", "val"))
    parser.add_argument("--train-split-ratio", type=float, default=1.0)
    parser.add_argument("--sampling-method", choices=("random", "stratified"), default="random")
    args = parser.parse_args()

    sys.path[:0] = [str(args.root / "src/CLEAR-HUG"), str(args.root / "mvp")]
    import modeling_finetune_fair_baselines  # noqa: F401
    from utils.QRSDataset import ECGDatasetFinetune

    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = create_model(
        "CLEAR_MASKED_DEEPSETS_finetune_base", pretrained=False,
        num_classes=args.classes, cls_token_num=12, padding_mask=False,
        atten_mask=True, mask_ratio=0,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    incompatible = model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
    model.cuda().eval()

    manifest_path = args.output / "feature-manifest.json"
    manifest = {
        "checkpoint": str(args.checkpoint), "checkpoint_seed": args.seed,
        "train_split_ratio": args.train_split_ratio,
        "sampling_method": args.sampling_method,
        "former_official_test_used": False, "splits": {},
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if existing.get("checkpoint") != str(args.checkpoint) or existing.get("checkpoint_seed") != args.seed:
            raise RuntimeError("Existing feature manifest uses a different checkpoint")
        manifest = existing

    with torch.inference_mode():
        for split in args.splits:
            split_output = args.output / split
            split_output.mkdir(exist_ok=True)
            ratio = args.train_split_ratio if split == "train" else 1.0
            dataset = ECGDatasetFinetune(
                str(args.dataset), stage=split, split_ratio=ratio,
                sampling_method=args.sampling_method,
            )
            loader = DataLoader(dataset, sampler=SequentialSampler(dataset),
                batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=False)
            total = len(dataset)
            local_out = open_array(split_output / "local.npy", (total, 15, 12, 768), np.float16)
            shared_out = open_array(split_output / "shared.npy", (total, 15, 768), np.float16)
            valid_out = open_array(split_output / "valid.npy", (total, 15, 12), np.bool_)
            logits_out = open_array(split_output / "baseline_logits.npy", (total, args.classes), np.float32)
            labels_out = open_array(split_output / "labels.npy", (total, args.classes), np.float32)
            offset = 0
            for signals, targets, in_chan, in_time, _ in loader:
                signals = signals.float().cuda(non_blocking=True)
                in_chan = in_chan.cuda(non_blocking=True)
                in_time = in_time.cuda(non_blocking=True)
                if model.backbone.norm_pix_loss:
                    mean = signals.mean(dim=-1, keepdim=True)
                    variance = signals.var(dim=-1, keepdim=True)
                    signals = (signals - mean) / torch.sqrt(variance + 1.0e-6)
                with torch.autocast("cuda", dtype=torch.float16):
                    encoded, _, _, ids_restore = model.backbone.forward_feature(
                        signals, mask_bool_matrix=None, key_padding_mask=None,
                        in_chan_matrix=in_chan, in_time_matrix=in_time, attn_mask=None)
                    lead_cls, shuffled = encoded[:, :12], encoded[:, 12:]
                    ordered = torch.gather(shuffled, 1, ids_restore.unsqueeze(-1).expand_as(shuffled))
                    batch = len(signals)
                    local = ordered.reshape(batch, 12, 15, 768).permute(0, 2, 1, 3)
                    valid = in_time.reshape(batch, 12, 15).gt(0).permute(0, 2, 1)
                    lead_valid = valid.any(dim=1)
                    summary = model.adapter(lead_cls, lead_valid)
                    baseline_logits = model.mlp_head(summary)
                    shared = model.adapter(local.reshape(batch * 15, 12, 768),
                        valid.reshape(batch * 15, 12)).reshape(batch, 15, 768)
                end = offset + batch
                local_out[offset:end] = local.float().cpu().numpy().astype(np.float16)
                shared_out[offset:end] = shared.float().cpu().numpy().astype(np.float16)
                valid_out[offset:end] = valid.cpu().numpy()
                logits_out[offset:end] = baseline_logits.float().cpu().numpy()
                labels_out[offset:end] = targets.numpy().astype(np.float32)
                offset = end
            for array in (local_out, shared_out, valid_out, logits_out, labels_out): array.flush()
            if offset != total: raise RuntimeError(f"{split}: wrote {offset}, expected {total}")
            manifest["splits"][split] = {"records": total, "path": str(split_output)}

    incoming = manifest_path.with_suffix(".json.incoming")
    incoming.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(incoming, manifest_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
