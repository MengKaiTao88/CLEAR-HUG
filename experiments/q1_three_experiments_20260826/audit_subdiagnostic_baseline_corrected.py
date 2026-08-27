"""Audit the development-only Subdiagnostic frozen-baseline cache.

The audit intentionally opens only train/val QRS arrays.  It verifies label
ordering, class metadata, split sizes, logit/probability semantics, and (when
CUDA is available) independently re-forwards the validation QRS through the
recorded CLEAR checkpoint to compare logits with the cached array.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
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


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    valid = [i for i in range(labels.shape[1]) if np.unique(labels[:, i]).size == 2]
    return {
        "macro_auroc": float(roc_auc_score(labels[:, valid], scores[:, valid], average="macro")),
        "macro_auprc": float(
            average_precision_score(labels[:, valid], scores[:, valid], average="macro")
        ),
        "valid_classes": len(valid),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_split(
    *,
    split: str,
    qrs_root: Path,
    labels_root: Path,
    cache_root: Path,
) -> dict:
    qrs = np.load(qrs_root / f"{split}_data.npy", mmap_mode="r")
    times = np.load(qrs_root / f"{split}_data_in_times.npy", mmap_mode="r")
    chans = np.load(qrs_root / f"{split}_data_in_chans.npy", mmap_mode="r")
    source_labels = np.load(labels_root / f"{split}_labels.npy", mmap_mode="r")
    cached_labels = np.load(cache_root / split / "labels.npy", mmap_mode="r")
    cached_logits = np.load(cache_root / split / "baseline_logits.npy", mmap_mode="r")
    if qrs.shape[0] != source_labels.shape[0] or qrs.shape[0] != times.shape[0]:
        raise RuntimeError(f"{split}: raw QRS/labels/times length mismatch")
    if not np.array_equal(source_labels, cached_labels):
        raise RuntimeError(f"{split}: cached labels differ from source labels")
    expected_channels = np.tile(
        np.repeat(np.arange(1, 13, dtype=chans.dtype), 15), (len(chans), 1)
    )
    if not np.array_equal(chans, expected_channels):
        raise RuntimeError(f"{split}: channel order is not lead-major 12x15")
    valid = [i for i in range(cached_labels.shape[1]) if np.unique(cached_labels[:, i]).size == 2]
    sigmoid = 1.0 / (1.0 + np.exp(-np.asarray(cached_logits, dtype=np.float32)))
    double_sigmoid = 1.0 / (1.0 + np.exp(-sigmoid))
    payload = {
        "split": split,
        "qrs_shape": list(qrs.shape),
        "times_shape": list(times.shape),
        "channels_shape": list(chans.shape),
        "source_labels_shape": list(source_labels.shape),
        "cached_labels_shape": list(cached_labels.shape),
        "cached_logits_shape": list(cached_logits.shape),
        "labels_equal_source": True,
        "channels_lead_major": True,
        "valid_classes": len(valid),
        "class_positive_counts": cached_labels.sum(axis=0).astype(int).tolist(),
        "time_positive_fraction": float((times > 0).mean()),
        "cached_logits_finite": bool(np.isfinite(cached_logits).all()),
        "cached_logits_min": float(cached_logits.min()),
        "cached_logits_max": float(cached_logits.max()),
        "raw_logits_metrics": _metrics(cached_labels, cached_logits),
        "sigmoid_metrics": _metrics(cached_labels, sigmoid),
        "double_sigmoid_metrics": _metrics(cached_labels, double_sigmoid),
        "source_labels_sha256": _sha256(labels_root / f"{split}_labels.npy"),
        "cached_labels_sha256": _sha256(cache_root / split / "labels.npy"),
    }
    path_file = labels_root / f"{split}_path.npy"
    if path_file.is_file():
        paths = np.load(path_file, mmap_mode="r")
        payload["source_paths_shape"] = list(paths.shape)
        payload["source_paths_sha256"] = _sha256(path_file)
    return payload


def _state_dict(payload):
    if isinstance(payload, dict):
        for key in ("model", "state_dict"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                return candidate
        if payload and all(hasattr(value, "numel") for value in payload.values()):
            return payload
    raise RuntimeError("checkpoint has no tensor state dictionary")


def _infer_baseline_model(state: dict, requested: str) -> str:
    if requested != "auto":
        return requested
    keys = tuple(state)
    if any(key.startswith("adapter.level1_experts") for key in keys):
        return "CLEAR_MASKED_HUG_finetune_base"
    if any(key.startswith("adapter.phi") for key in keys):
        return "CLEAR_MASKED_DEEPSETS_finetune_base"
    # The released legacy HUG checkpoint has a stage attribute in the model
    # definition and only contains backbone/mlp_head weights.
    return "CLEAR_HUG_finetune_base"


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def _aggregate_logits(model, model_name: str, lead_cls: torch.Tensor, valid: torch.Tensor):
    """Apply the selected frozen record-level aggregation exactly once."""
    if model_name == "CLEAR_HUG_finetune_base":
        # The legacy HUG stage-2 checkpoint routes Lead-CLS tokens through
        # its learned seven-group MoE before the record head.  A plain mean
        # here silently recreates the invalid historical cache and defeats
        # this audit.
        if not hasattr(model, "backbone") or not hasattr(model.backbone, "moe"):
            raise RuntimeError("legacy HUG model has no backbone.moe aggregation")
        groups = model.backbone.moe(lead_cls)
        summary = torch.stack(groups, dim=1).mean(dim=1)
        return model.mlp_head(summary)
    if model_name == "CLEAR_MASKED_DEEPSETS_finetune_base":
        return model.mlp_head(model.adapter(lead_cls, valid.any(dim=1)))
    if model_name == "CLEAR_MASKED_HUG_finetune_base":
        from modeling_finetune_fair_baselines import masked_mean

        groups, group_valid = model.adapter(lead_cls, valid.any(dim=1))
        summary, _ = masked_mean(groups, group_valid, dim=1)
        return model.mlp_head(summary)
    raise RuntimeError(f"unsupported baseline model {model_name}")


def _forward_validation(
    *, root: Path, qrs_root: Path, labels_root: Path, cache_root: Path, released_checkpoint: Path,
    baseline_checkpoint: Path, baseline_model_name: str, batch_size: int, split: str,
) -> dict:
    """Recompute one development split and compare cache without old summaries."""
    sys_path = str(root / "src/CLEAR-HUG")
    mvp_path = str(root / "mvp")

    for path in (sys_path, mvp_path):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        from clear_adapter import ClearFeatureAdapter
        import modeling_finetune_fair_baselines  # noqa: F401
        import modeling_finetune  # noqa: F401 (registers legacy HUG)
        checkpoint_payload = torch.load(baseline_checkpoint, map_location="cpu")
        state = _state_dict(checkpoint_payload)
        model = create_model(
            baseline_model_name,
            pretrained=False,
            num_classes=23,
            cls_token_num=12,
            padding_mask=False,
            atten_mask=True,
            mask_ratio=0,
        )
        incompatible = model.load_state_dict(state, strict=False)
        feature_model = ClearFeatureAdapter(root / "src/CLEAR-HUG", released_checkpoint)
        preferred = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = model.to(preferred).eval()
            feature_model = feature_model.to(preferred).eval()
            device = preferred
            with torch.inference_mode():
                dataset = QRSDevelopmentDataset(qrs_root, split, labels_root)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                outputs = []
                labels = []
                started = time.perf_counter()
                for signals, batch_labels, channels, times in loader:
                    signals = signals.to(device, non_blocking=True).float()
                    channels = channels.to(device, non_blocking=True)
                    times = times.to(device, non_blocking=True)
                    with _autocast(device):
                        local, lead_cls = feature_model(signals, channels, times)
                        batch = signals.shape[0]
                        valid = times.reshape(batch, 12, 15).gt(0).permute(0, 2, 1)
                        logits = _aggregate_logits(model, baseline_model_name, lead_cls, valid)
                    outputs.append(logits.float().cpu().numpy())
                    labels.append(batch_labels.numpy())
                recomputed = np.concatenate(outputs, axis=0)
                truth = np.concatenate(labels, axis=0)
                cached = np.load(cache_root / split / "baseline_logits.npy", mmap_mode="r")
                cached_labels = np.load(cache_root / split / "labels.npy", mmap_mode="r")
                diff = np.asarray(recomputed, dtype=np.float32) - np.asarray(cached, dtype=np.float32)
                result = {
                    "performed": True,
                    "split": split,
                    "device": str(device),
                    "baseline_model": baseline_model_name,
                    "checkpoint_missing_keys": list(incompatible.missing_keys),
                    "checkpoint_unexpected_keys": list(incompatible.unexpected_keys),
                    "recomputed_shape": list(recomputed.shape),
                    "labels_match_cache": bool(np.array_equal(truth, cached_labels)),
                    "max_abs_diff": float(np.abs(diff).max()),
                    "mean_abs_diff": float(np.abs(diff).mean()),
                    "allclose_atol_1e-5": bool(np.allclose(recomputed, cached, atol=1e-5, rtol=1e-5)),
                    "recomputed_metrics": _metrics(truth, recomputed),
                    "cached_metrics": _metrics(np.asarray(cached_labels), np.asarray(cached)),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                return result
        except Exception as error:
            # A mismatched driver (notably CUDA error 804) must be recorded,
            # then retried on CPU when possible rather than being mistaken for
            # a metric mismatch.
            if preferred.type != "cuda":
                raise
            try:
                model = model.to("cpu").eval()
                feature_model = feature_model.to("cpu").eval()
                device = torch.device("cpu")
                with torch.inference_mode():
                    dataset = QRSDevelopmentDataset(qrs_root, split, labels_root)
                    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                    outputs, labels = [], []
                    for signals, batch_labels, channels, times in loader:
                        with _autocast(device):
                            local, lead_cls = feature_model(signals.float(), channels, times)
                            batch = signals.shape[0]
                            valid = times.reshape(batch, 12, 15).gt(0).permute(0, 2, 1)
                            logits = _aggregate_logits(model, baseline_model_name, lead_cls, valid)
                        outputs.append(logits.float().numpy())
                        labels.append(batch_labels.numpy())
                recomputed = np.concatenate(outputs, axis=0)
                truth = np.concatenate(labels, axis=0)
                cached = np.load(cache_root / split / "baseline_logits.npy", mmap_mode="r")
                cached_labels = np.load(cache_root / split / "labels.npy", mmap_mode="r")
                diff = np.asarray(recomputed, dtype=np.float32) - np.asarray(cached, dtype=np.float32)
                return {
                    "performed": True,
                    "split": split,
                    "device": "cpu-fallback",
                    "cuda_error": f"{type(error).__name__}: {error}",
                    "baseline_model": baseline_model_name,
                    "checkpoint_missing_keys": list(incompatible.missing_keys),
                    "checkpoint_unexpected_keys": list(incompatible.unexpected_keys),
                    "recomputed_shape": list(recomputed.shape),
                    "labels_match_cache": bool(np.array_equal(truth, cached_labels)),
                    "max_abs_diff": float(np.abs(diff).max()),
                    "mean_abs_diff": float(np.abs(diff).mean()),
                    "allclose_atol_1e-5": bool(np.allclose(recomputed, cached, atol=1e-5, rtol=1e-5)),
                    "recomputed_metrics": _metrics(truth, recomputed),
                    "cached_metrics": _metrics(np.asarray(cached_labels), np.asarray(cached)),
                }
            except Exception as fallback_error:
                return {
                    "performed": False,
                    "baseline_model": baseline_model_name,
                    "cuda_error": f"{type(error).__name__}: {error}",
                    "cpu_fallback_error": f"{type(fallback_error).__name__}: {fallback_error}",
                }
    except Exception as error:
        return {"performed": False, "baseline_model": baseline_model_name, "error": f"{type(error).__name__}: {error}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--qrs-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--released-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--baseline-model",
        default="auto",
        choices=(
            "auto",
            "CLEAR_HUG_finetune_base",
            "CLEAR_MASKED_HUG_finetune_base",
            "CLEAR_MASKED_DEEPSETS_finetune_base",
        ),
    )
    parser.add_argument("--forward-split", choices=("train", "val"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    for path in (args.qrs_root, args.labels_root, args.cache_root, args.released_checkpoint, args.baseline_checkpoint):
        if "test" in {part.lower() for part in path.parts}:
            raise RuntimeError(f"development-only audit refuses test path: {path}")
    with (args.labels_root / "mlb.pkl").open("rb") as handle:
        mlb = pickle.load(handle)
    checkpoint_payload = torch.load(args.baseline_checkpoint, map_location="cpu")
    checkpoint_state = _state_dict(checkpoint_payload)
    baseline_model_name = _infer_baseline_model(checkpoint_state, args.baseline_model)
    payload = {
        "schema_version": 1,
        "scope": "development train/validation only",
        "formal_test_used": False,
        "task": "PTBXL Subdiagnostic",
        "classes": list(mlb.classes_),
        "splits": {
            split: _audit_split(
                split=split,
                qrs_root=args.qrs_root,
                labels_root=args.labels_root,
                cache_root=args.cache_root,
            )
            for split in ("train", "val")
        },
        "checkpoint": str(args.baseline_checkpoint),
        "baseline_model_requested": args.baseline_model,
        "baseline_model_inferred": baseline_model_name,
        "checkpoint_state_keys_sample": sorted(checkpoint_state)[:20],
    }
    payload["independent_validation_forward"] = _forward_validation(
        root=args.root,
        qrs_root=args.qrs_root,
        labels_root=args.labels_root,
        cache_root=args.cache_root,
        released_checkpoint=args.released_checkpoint,
        baseline_checkpoint=args.baseline_checkpoint,
        baseline_model_name=baseline_model_name,
        batch_size=args.batch_size,
        split=args.forward_split,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

