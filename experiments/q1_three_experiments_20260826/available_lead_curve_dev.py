"""Validation-only available-lead curve for the CLEAR aggregation models.

The released CLEAR encoder is run once per validation batch.  HUG and
DeepSets use their learned lead aggregation heads, while HiLAR uses the
frozen DeepSets logits plus the released direct residual checkpoint.  Lead
subsets are sampled from the twelve observed leads; no test array is opened.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


def add_paths(root: Path) -> None:
    sys.path[:0] = [str(root / "src/CLEAR-HUG"), str(root / "mvp")]


class ValidationQRS(Dataset):
    """Open only the validation arrays and reconstruct missing channel IDs."""

    def __init__(self, dataset: Path) -> None:
        self.data = np.load(dataset / "val_data.npy", mmap_mode="r")
        self.labels = np.load(dataset / "val_labels.npy", mmap_mode="r")
        self.times = np.load(dataset / "val_data_in_times.npy", mmap_mode="r")
        channel_path = dataset / "val_data_in_chans.npy"
        if channel_path.is_file():
            self.channels = np.load(channel_path, mmap_mode="r")
        else:
            self.channels = np.tile(
                np.repeat(np.arange(1, 13, dtype=np.int64), 15),
                (len(self.data), 1),
            )
        if not (len(self.data) == len(self.labels) == len(self.times)):
            raise RuntimeError("validation arrays have inconsistent lengths")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            np.asarray(self.data[index], dtype=np.float32),
            np.asarray(self.labels[index], dtype=np.float32),
            np.asarray(self.channels[index], dtype=np.int64),
            np.asarray(self.times[index], dtype=np.int64),
        )


def load_state(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    return payload.get("model", payload) if isinstance(payload, dict) else payload


def load_fair_model(root: Path, name: str, checkpoint: Path, classes: int):
    from timm.models import create_model

    # The module import registers the fair aggregation model names.
    import modeling_finetune_fair_baselines  # noqa: F401

    model = create_model(
        name,
        pretrained=False,
        num_classes=classes,
        cls_token_num=12,
        padding_mask=False,
        atten_mask=True,
        mask_ratio=0,
    )
    incompatible = model.load_state_dict(load_state(checkpoint), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"{name} checkpoint mismatch: missing={incompatible.missing_keys[:8]} "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )
    return model


def masked_mean(values: torch.Tensor, valid: torch.Tensor, dim: int):
    from modeling_finetune_fair_baselines import masked_mean as fair_masked_mean

    return fair_masked_mean(values, valid, dim=dim)


def aggregate_hug(model, lead_cls: torch.Tensor, lead_valid: torch.Tensor):
    groups, group_valid = model.adapter(lead_cls, lead_valid)
    summary, _ = masked_mean(groups, group_valid, dim=1)
    return model.mlp_head(summary)


def aggregate_deepsets(model, lead_cls: torch.Tensor, lead_valid: torch.Tensor):
    summary = model.adapter(lead_cls, lead_valid)
    return model.mlp_head(summary)


def macro_metrics(truth: np.ndarray, score: np.ndarray) -> dict[str, float]:
    valid = [
        column
        for column in range(truth.shape[1])
        if np.unique(truth[:, column]).size == 2
    ]
    return {
        "macro_auroc": float(roc_auc_score(truth[:, valid], score[:, valid], average="macro")),
        "macro_auprc": float(
            average_precision_score(truth[:, valid], score[:, valid], average="macro")
        ),
        "valid_classes": len(valid),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--released-checkpoint", type=Path, required=True)
    parser.add_argument("--hug-checkpoint", type=Path, required=True)
    parser.add_argument("--deepsets-checkpoint", type=Path, required=True)
    parser.add_argument("--hilar-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--masks-per-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    if "test" in {part.lower() for part in args.dataset.parts}:
        raise ValueError("test paths are forbidden")

    add_paths(args.root)
    from clear_adapter import ClearFeatureAdapter
    from modeling_deepsets_anchored_residual import AnchoredResidualClassifier

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this validation benchmark")
    device = torch.device("cuda")

    encoder = ClearFeatureAdapter(args.root / "src/CLEAR-HUG", args.released_checkpoint).to(device).eval()
    hug = load_fair_model(
        args.root,
        "CLEAR_MASKED_HUG_finetune_base",
        args.hug_checkpoint,
        args.classes,
    ).to(device).eval()
    deepsets = load_fair_model(
        args.root,
        "CLEAR_MASKED_DEEPSETS_finetune_base",
        args.deepsets_checkpoint,
        args.classes,
    ).to(device).eval()
    hilar = AnchoredResidualClassifier(args.classes, "direct").to(device).eval()
    hilar.load_state_dict(load_state(args.hilar_checkpoint), strict=True)

    lead_subsets: dict[int, list[list[int]]] = {}
    rng = np.random.default_rng(args.seed)
    for count in (2, 4, 6, 8, 10, 12):
        if count == 12:
            lead_subsets[count] = [list(range(12))]
        else:
            masks = {tuple(sorted(rng.choice(12, count, replace=False).tolist())) for _ in range(args.masks_per_k)}
            lead_subsets[count] = [list(mask) for mask in sorted(masks)]

    loader = DataLoader(
        ValidationQRS(args.dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    scores: dict[tuple[int, int, str], list[np.ndarray]] = {}
    truths: list[np.ndarray] = []
    with torch.inference_mode():
        for signals, labels, channels, times in loader:
            signals = signals.to(device, non_blocking=True)
            channels = channels.to(device, non_blocking=True)
            times = times.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                local, lead_cls = encoder(signals, channels, times)
            batch = len(signals)
            valid = times.reshape(batch, 12, 15).gt(0).permute(0, 2, 1)
            truths.append(labels.numpy())
            for count, masks in lead_subsets.items():
                for mask_index, mask in enumerate(masks):
                    keep = torch.zeros(12, dtype=torch.bool, device=device)
                    keep[mask] = True
                    masked_valid = valid & keep[None, None, :]
                    lead_valid = masked_valid.any(dim=1)
                    with torch.autocast("cuda", dtype=torch.float16):
                        hug_logits = aggregate_hug(hug, lead_cls, lead_valid)
                        deep_logits = aggregate_deepsets(deepsets, lead_cls, lead_valid)
                        masked_local = local * keep[None, None, :, None].to(local.dtype)
                        zero_shared = torch.zeros(
                            batch, 15, 768, device=device, dtype=local.dtype
                        )
                        hilar_logits, _ = hilar(
                            masked_local, zero_shared, masked_valid, deep_logits
                        )
                    outputs = {
                        "HUG": torch.sigmoid(hug_logits).float().cpu().numpy(),
                        "DeepSets": torch.sigmoid(deep_logits).float().cpu().numpy(),
                        "HiLAR / Final Direct": torch.sigmoid(hilar_logits).float().cpu().numpy(),
                    }
                    for model_name, output in outputs.items():
                        scores.setdefault((count, mask_index, model_name), []).append(output)

    truth = np.concatenate(truths)
    results: dict[str, dict] = {}
    for count, masks in lead_subsets.items():
        per_model: dict[str, dict] = {}
        for model_name in ("HUG", "DeepSets", "HiLAR / Final Direct"):
            rows = []
            for mask_index, mask in enumerate(masks):
                score = np.concatenate(scores[(count, mask_index, model_name)])
                rows.append({"mask": mask, **macro_metrics(truth, score)})
            per_model[model_name] = {
                "mask_count": len(rows),
                "macro_auroc_mean": float(np.mean([row["macro_auroc"] for row in rows])),
                "macro_auroc_std": float(np.std([row["macro_auroc"] for row in rows])),
                "macro_auprc_mean": float(np.mean([row["macro_auprc"] for row in rows])),
                "macro_auprc_std": float(np.std([row["macro_auprc"] for row in rows])),
                "masks": rows,
            }
        results[str(count)] = per_model

    payload = {
        "schema_version": 1,
        "scope": "development validation only",
        "formal_test_used": False,
        "task": "PTB-XL Superdiagnostic",
        "dataset": str(args.dataset),
        "split": "val",
        "records": int(len(truth)),
        "classes": int(args.classes),
        "seed": args.seed,
        "masks_per_k_requested": args.masks_per_k,
        "lead_counts": [2, 4, 6, 8, 10, 12],
        "models": {
            "HUG": str(args.hug_checkpoint),
            "DeepSets": str(args.deepsets_checkpoint),
            "HiLAR / Final Direct": str(args.hilar_checkpoint),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    incoming = args.output.with_suffix(args.output.suffix + ".incoming")
    incoming.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    incoming.replace(args.output)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
