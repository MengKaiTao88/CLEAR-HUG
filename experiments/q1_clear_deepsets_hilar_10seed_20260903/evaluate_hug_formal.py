"""Fresh formal-test inference for one CLEAR-HUG checkpoint."""

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
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_dict(payload):
    if isinstance(payload, dict):
        for key in ("model", "state_dict"):
            if isinstance(payload.get(key), dict):
                return payload[key]
        if payload and all(hasattr(value, "numel") for value in payload.values()):
            return payload
    raise RuntimeError("checkpoint has no tensor state dictionary")


def metrics(truth: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    valid = [i for i in range(truth.shape[1]) if np.unique(truth[:, i]).size == 2]
    return {
        "macro_auroc": float(roc_auc_score(truth[:, valid], score[:, valid], average="macro")),
        "macro_auprc": float(average_precision_score(truth[:, valid], score[:, valid], average="macro")),
        "valid_classes": len(valid),
    }


class TestDataset(Dataset):
    def __init__(self, root: Path):
        self.data = np.load(root / "test_data.npy", mmap_mode="r")
        self.labels = np.load(root / "test_labels.npy", mmap_mode="r")
        self.times = np.load(root / "test_data_in_times.npy", mmap_mode="r")
        channel_path = root / "test_data_in_chans.npy"
        self.channels = np.load(channel_path, mmap_mode="r") if channel_path.exists() else np.tile(
            np.repeat(np.arange(1, 13, dtype=np.int64), 15), (len(self.data), 1)
        )
        if self.data.shape[1:] != (180, 96) or len(self.data) != len(self.labels):
            raise RuntimeError("invalid formal-test arrays")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return tuple(np.asarray(value[index]) for value in (self.data, self.labels, self.channels, self.times))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--released-checkpoint", type=Path, required=True)
    parser.add_argument("--strict-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    if args.seed not in range(42, 62) or not torch.cuda.is_available():
        raise RuntimeError("invalid seed or CUDA unavailable")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    sys.path[:0] = [str(args.root / "src/CLEAR-HUG"), str(args.root / "mvp")]
    import modeling_finetune  # noqa: F401
    from clear_adapter import ClearFeatureAdapter
    from timm.models import create_model

    dataset = TestDataset(args.dataset)
    strict = np.load(args.strict_labels, mmap_mode="r")
    if strict.shape != dataset.labels.shape or not np.array_equal(strict, dataset.labels):
        raise RuntimeError("HUG labels/order differ from paired DeepSets labels")
    model = create_model("CLEAR_HUG_finetune_base", pretrained=False, num_classes=args.classes,
                         cls_token_num=12, padding_mask=False, atten_mask=True, mask_ratio=0)
    incompatible = model.load_state_dict(state_dict(torch.load(args.checkpoint, map_location="cpu")), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys or not hasattr(model.backbone, "moe"):
        raise RuntimeError(f"HUG checkpoint mismatch: {incompatible}")
    model.cuda().eval()
    feature_model = ClearFeatureAdapter(args.root / "src/CLEAR-HUG", args.released_checkpoint).cuda().eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=6, pin_memory=True)
    truth_rows, logit_rows = [], []
    with torch.inference_mode():
        for signals, labels, channels, times in loader:
            with torch.autocast("cuda", dtype=torch.float16):
                _, lead_cls = feature_model(signals.cuda().float(), channels.cuda(), times.cuda())
                groups = model.backbone.moe(lead_cls)
                if len(groups) != 7:
                    raise RuntimeError(f"expected seven HUG groups, got {len(groups)}")
                logits = model.mlp_head(torch.stack(groups, dim=1).mean(dim=1))
            truth_rows.append(labels.numpy().astype(np.float32, copy=False))
            logit_rows.append(logits.float().cpu().numpy())
    truth = np.concatenate(truth_rows); logits = np.concatenate(logit_rows)
    score = torch.sigmoid(torch.from_numpy(logits.astype(np.float64))).numpy()
    if not np.array_equal(truth, strict):
        raise RuntimeError("DataLoader changed formal-test label order")
    payload = {
        "schema_version": 1, "status": "complete", "task": args.task, "seed": args.seed,
        "model": "CLEAR_HUG_finetune_base", "aggregation": "seven moe groups -> mean -> mlp_head",
        "metrics": metrics(truth, score), "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint), "test_used_for_selection": False,
        "archived_predictions_used": False,
    }
    args.output.mkdir(parents=True)
    predictions = args.output / "test_predictions.npz"
    np.savez_compressed(predictions, y_true=truth, hug_score=score, hug_logits=logits)
    payload["predictions_sha256"] = sha256(predictions)
    incoming = args.output / "formal-test-result.json.incoming"
    incoming.write_text(json.dumps(payload, indent=2) + "\n"); os.replace(incoming, args.output / "formal-test-result.json")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
