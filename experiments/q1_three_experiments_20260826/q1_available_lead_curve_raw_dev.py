"""Development-only available-lead curve with a strict k=12 gate.

This benchmark intentionally follows the frozen missing-lead intervention:
raw ECG leads are zeroed, the official QRS tokenizer is rerun, and the
standard CLEAR ``forward_feature`` path is used.  Only validation arrays are
opened.  The first invocation must run ``--counts 12``.  It creates a clean
QRS reference, checks that the tokenized k=12 path reproduces standard
validation logits, and writes a gate marker.  A later invocation for
``--counts 2 4 6 8 10`` refuses to run unless that gate passed.

The script is deliberately separate from ``available_lead_curve_dev.py``:
that earlier probe masks deterministic encoded features after the encoder and
therefore cannot be compared with the raw-waveform formal protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


LEAD_COUNTS = (2, 4, 6, 8, 10, 12)
DEFAULT_MASK_SEED = 20260828
DEFAULT_BATCH_SIZE = 384
DEFAULT_WORKERS = 8
GATE_TOL = 1.0e-4


def reject_test_path(path: Path) -> None:
    """Reject paths whose components indicate the formal test split."""

    lowered = {part.lower() for part in path.parts}
    if "test" in lowered or any("test" in part for part in lowered):
        raise ValueError(f"test paths are forbidden: {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = path.with_suffix(path.suffix + ".incoming")
    incoming.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(incoming, path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_raw(raw: np.ndarray) -> np.ndarray:
    if raw.ndim != 3:
        raise ValueError(f"expected rank-3 raw ECG, got {raw.shape}")
    if raw.shape[1] == 12:
        return raw
    if raw.shape[2] == 12:
        return raw.transpose(0, 2, 1)
    raise ValueError(f"cannot identify 12-lead axis in {raw.shape}")


def resolve_clear_repo() -> Path:
    """Resolve the CLEAR source tree for both mvp and tracked script layouts."""

    configured_root = os.environ.get("CLEAR_HUG_ROOT")
    if configured_root:
        candidate = Path(configured_root) / "src" / "CLEAR-HUG"
        if (candidate / "QRSTokenizer.py").is_file():
            return candidate
    here = Path(__file__).resolve()
    candidates = (
        here.parents[2] if len(here.parents) > 2 else here.parent,
        here.parents[1] / "src" / "CLEAR-HUG" if len(here.parents) > 1 else here.parent,
    )
    for candidate in candidates:
        if (candidate / "QRSTokenizer.py").is_file():
            return candidate
    raise FileNotFoundError("could not resolve CLEAR-HUG source tree")


class ValidationQRS(Dataset):
    """Validation-only QRS dataset matching ``ECGDatasetFinetune`` order."""

    def __init__(self, qrs_dir: Path):
        reject_test_path(qrs_dir)
        self.data = np.load(qrs_dir / "val_data.npy", mmap_mode="r")
        self.labels = np.load(qrs_dir / "val_labels.npy", mmap_mode="r")
        self.channels = np.load(qrs_dir / "val_data_in_chans.npy", mmap_mode="r")
        self.times = np.load(qrs_dir / "val_data_in_times.npy", mmap_mode="r")
        self.pad = np.load(qrs_dir / "val_data_mask_pad.npy", mmap_mode="r")
        lengths = {len(self.data), len(self.labels), len(self.channels), len(self.times), len(self.pad)}
        if lengths != {len(self.data)}:
            raise RuntimeError(f"validation QRS arrays are not aligned: {lengths}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            np.asarray(self.data[index], dtype=np.float32),
            np.asarray(self.labels[index], dtype=np.float32),
            np.asarray(self.channels[index], dtype=np.int64),
            np.asarray(self.times[index], dtype=np.int64),
            np.asarray(self.pad[index], dtype=np.int64),
        )


def make_masks(mask_seed: int, masks_per_k: int) -> dict[str, list[list[int]]]:
    rng = np.random.default_rng(mask_seed)
    masks: dict[str, list[list[int]]] = {"12": [list(range(12))]}
    for count in LEAD_COUNTS[:-1]:
        selected: set[tuple[int, ...]] = set()
        while len(selected) < masks_per_k:
            selected.add(tuple(sorted(rng.choice(12, count, replace=False).tolist())))
        masks[str(count)] = [list(item) for item in sorted(selected)]
    return masks


def load_state(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    return payload.get("model", payload) if isinstance(payload, dict) else payload


def load_fair_model(root: Path, name: str, checkpoint: Path, classes: int):
    del root
    from timm.models import create_model

    # Import registers the controlled aggregation model names.
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


def macro_metrics(truth: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    valid = [
        index for index in range(truth.shape[1]) if np.unique(truth[:, index]).size == 2
    ]
    return {
        "macro_auroc": float(roc_auc_score(truth[:, valid], score[:, valid], average="macro")),
        "macro_auprc": float(
            average_precision_score(truth[:, valid], score[:, valid], average="macro")
        ),
        "valid_classes": len(valid),
    }


def make_qrs_variant(
    raw: np.ndarray,
    source_labels: np.ndarray,
    source_paths: np.ndarray,
    target: Path,
    keep: list[int],
    variant: str,
) -> dict[str, Any]:
    """Zero raw leads, rerun the official tokenizer, and write val-only QRS."""

    marker = target / "complete.json"
    if marker.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("variant") != variant or payload.get("records") != len(raw):
            raise RuntimeError(f"incompatible completion marker: {marker}")
        return payload

    clear_repo = resolve_clear_repo()
    sys.path.insert(0, str(clear_repo))
    from QRSTokenizer import QRSTokenizer

    incoming = target.with_name(target.name + ".incoming")
    if incoming.exists():
        shutil.rmtree(incoming)
    incoming.mkdir(parents=True)
    (incoming / "figs").mkdir()

    data = np.asarray(raw, dtype=np.float32).copy()
    absent = np.ones(12, dtype=bool)
    absent[np.asarray(keep, dtype=np.int64)] = False
    data[:, absent, :] = 0.0

    tokenizer = QRSTokenizer(
        fs=100,
        max_len=180,
        token_len=96,
        lead_len=15,
        save_path=str(incoming),
        stage="val",
        used_chans=list(range(12)),
    )
    tokenizer(data, plot=False)

    token_path = incoming / "val_data.npy"
    emitted = np.load(token_path, mmap_mode="r")
    if len(emitted) != len(raw):
        raise RuntimeError(
            f"{variant}: tokenizer emitted {len(emitted)} rows, expected {len(raw)}"
        )
    for name in (
        "val_data_in_chans.npy",
        "val_data_in_times.npy",
        "val_data_mask_pad.npy",
    ):
        if not (incoming / name).is_file():
            raise RuntimeError(f"tokenizer did not emit {incoming / name}")

    in_times = np.load(incoming / "val_data_in_times.npy")
    pad_mask = np.load(incoming / "val_data_mask_pad.npy")
    token_absent = np.repeat(absent, 15)
    in_times[:, token_absent] = 0
    pad_mask[:, token_absent] = 1
    np.save(incoming / "val_data_in_times.npy", in_times)
    np.save(incoming / "val_data_mask_pad.npy", pad_mask)
    np.save(incoming / "val_missing_lead_mask.npy", np.broadcast_to(absent, (len(raw), 12)))
    np.save(incoming / "val_labels.npy", source_labels)
    np.save(incoming / "val_path.npy", source_paths)

    payload = {
        "schema_version": 1,
        "scope": "development validation only",
        "formal_test_used": False,
        "variant": variant,
        "records": int(len(raw)),
        "kept_leads_zero_based": list(keep),
        "missing_leads_zero_based": np.flatnonzero(absent).tolist(),
        "raw_intervention": "absent leads set to zero before released QRS tokenizer",
        "explicit_mask": "in_times=0 and mask_pad=1 for every absent-lead token",
        "source_labels_sha256": hashlib.sha256(source_labels.tobytes()).hexdigest(),
        "source_paths_sha256": hashlib.sha256(source_paths.tobytes()).hexdigest(),
    }
    json_dump(incoming / "complete.json", payload)
    incoming.rename(target)
    return payload


def qrs_signature(qrs_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in (
        "val_data.npy",
        "val_data_in_chans.npy",
        "val_data_in_times.npy",
        "val_data_mask_pad.npy",
        "val_labels.npy",
        "val_path.npy",
    ):
        path = qrs_dir / name
        # The clean QRS bundle predates persisted path arrays and only keeps
        # the labels symlink.  The raw validation path array is used for row
        # alignment in that case; masked variants always write val_path.npy.
        if not path.is_file():
            continue
        result[name] = {
            "path": str(path),
            "sha256": sha256(path),
            "shape": list(np.load(path, mmap_mode="r").shape),
        }
    return result


def load_existing_standard(path: Path, truth: np.ndarray) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False}
    payload = np.load(path, allow_pickle=False)
    y_true = np.asarray(payload["y_true"])
    y_score = np.asarray(payload["y_score"])
    if not np.array_equal(y_true, truth):
        raise RuntimeError(f"stored standard validation labels differ: {path}")
    return {
        "available": True,
        "path": str(path),
        "records": int(len(y_true)),
        "score": y_score,
        "sha256": sha256(path),
    }


def _loader(qrs_dir: Path, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        ValidationQRS(qrs_dir),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


@torch.inference_mode()
def predict_fair(
    root: Path,
    qrs_dir: Path,
    model_name: str,
    checkpoint: Path,
    classes: int,
    seed: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, np.ndarray]:
    """Run the same standard full-model forward used by export_split_predictions."""

    reject_test_path(qrs_dir)
    seed_all(seed)
    model = load_fair_model(root, model_name, checkpoint, classes).cuda().eval()
    truth: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for signals, labels, channels, times, pad in _loader(qrs_dir, batch_size, num_workers):
        del pad
        signals = signals.float().cuda(non_blocking=True)
        channels = channels.cuda(non_blocking=True)
        times = times.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(
                signals,
                in_chan_matrix=channels,
                in_time_matrix=times,
                key_padding_mask=None,
                attn_mask=None,
            )
        truth.append(labels.numpy().astype(np.float32, copy=False))
        logits.append(output.float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return {"truth": np.concatenate(truth), "logits": np.concatenate(logits)}


@torch.inference_mode()
def predict_hilar(
    root: Path,
    qrs_dir: Path,
    deepsets_checkpoint: Path,
    hilar_checkpoint: Path,
    classes: int,
    seed: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, np.ndarray]:
    """Run HiLAR using official ``forward_feature`` and ordered restoration."""

    reject_test_path(qrs_dir)
    seed_all(seed)
    base = load_fair_model(
        root, "CLEAR_MASKED_DEEPSETS_finetune_base", deepsets_checkpoint, classes
    ).cuda().eval()
    from modeling_deepsets_anchored_residual import AnchoredResidualClassifier

    residual = AnchoredResidualClassifier(classes, "direct").cuda().eval()
    residual.load_state_dict(load_state(hilar_checkpoint), strict=True)
    truth: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for signals, labels, channels, times, pad in _loader(qrs_dir, batch_size, num_workers):
        del pad
        signals = signals.float().cuda(non_blocking=True)
        channels = channels.cuda(non_blocking=True)
        times = times.cuda(non_blocking=True)
        batch = len(signals)
        with torch.autocast("cuda", dtype=torch.float16):
            # Match CLEARFairAggregation.forward exactly.  The residual path
            # calls the backbone directly, so it must apply the released
            # per-record normalization before forward_feature itself.
            if base.backbone.norm_pix_loss:
                mean = signals.mean(dim=-1, keepdim=True)
                var = signals.var(dim=-1, keepdim=True)
                signals = (signals - mean) / torch.sqrt(var + 1.0e-6)
            # This is deliberately the official stochastic CLEAR path.  The
            # ids_restore gather recreates the original lead-major local grid.
            encoded, _, _, ids_restore = base.backbone.forward_feature(
                signals,
                mask_bool_matrix=None,
                key_padding_mask=None,
                in_chan_matrix=channels,
                in_time_matrix=times,
                attn_mask=None,
            )
            lead_cls = encoded[:, :12]
            shuffled = encoded[:, 12:]
            ordered = torch.gather(
                shuffled,
                1,
                ids_restore.unsqueeze(-1).expand_as(shuffled),
            )
            local = ordered.reshape(batch, 12, 15, 768).permute(0, 2, 1, 3)
            valid = times.reshape(batch, 12, 15).gt(0).permute(0, 2, 1)
            lead_valid = valid.any(dim=1)
            baseline_summary = base.adapter(lead_cls, lead_valid)
            baseline_logits = base.mlp_head(baseline_summary)
            flat_local = local.reshape(batch * 15, 12, 768)
            flat_valid = valid.reshape(batch * 15, 12)
            shared = base.adapter(flat_local, flat_valid).reshape(batch, 15, 768)
            output, delta = residual(local, shared, valid, baseline_logits)
        truth.append(labels.numpy().astype(np.float32, copy=False))
        logits.append(output.float().cpu().numpy())
    del residual, base
    torch.cuda.empty_cache()
    return {"truth": np.concatenate(truth), "logits": np.concatenate(logits)}


def save_prediction(path: Path, prediction: dict[str, np.ndarray], mask: list[int], seed: int, model: str) -> dict[str, Any]:
    truth = prediction["truth"]
    logits = prediction["logits"]
    score = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    metrics = macro_metrics(truth, score)
    np.savez_compressed(path, y_true=truth, logits=logits, y_score=score, mask=np.asarray(mask), seed=np.asarray(seed), model=np.asarray(model))
    return {"model": model, "seed": seed, "mask": mask, **metrics, "path": str(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--clean-qrs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hug-checkpoint", type=Path, required=True)
    parser.add_argument("--deepsets-checkpoint", type=Path, required=True)
    parser.add_argument("--hilar-checkpoint", type=Path, required=True)
    parser.add_argument("--standard-deepsets-predictions", type=Path, default=None)
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-seed", type=int, default=DEFAULT_MASK_SEED)
    parser.add_argument("--masks-per-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--counts", nargs="+", type=int, required=True)
    parser.add_argument("--tol", type=float, default=GATE_TOL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(count not in LEAD_COUNTS for count in args.counts):
        raise ValueError(f"counts must be in {LEAD_COUNTS}")
    if len(set(args.counts)) != len(args.counts):
        raise ValueError("duplicate counts")
    reject_test_path(args.raw_dir)
    reject_test_path(args.clean_qrs)
    reject_test_path(args.output)
    for path in (args.raw_dir / "val_data.npy", args.raw_dir / "val_labels.npy", args.raw_dir / "val_path.npy"):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (
        args.clean_qrs / "val_data.npy",
        args.clean_qrs / "val_labels.npy",
        args.clean_qrs / "val_data_in_chans.npy",
        args.clean_qrs / "val_data_in_times.npy",
        args.clean_qrs / "val_data_mask_pad.npy",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (args.hug_checkpoint, args.deepsets_checkpoint, args.hilar_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    args.output.mkdir(parents=True, exist_ok=True)
    masks_path = args.output / "masks.json"
    masks = make_masks(args.mask_seed, args.masks_per_k)
    if masks_path.is_file():
        existing = json.loads(masks_path.read_text(encoding="utf-8"))
        if existing.get("mask_seed") != args.mask_seed or existing.get("masks") != masks:
            raise RuntimeError("existing masks.json differs from requested fixed masks")
    else:
        json_dump(
            masks_path,
            {
                "schema_version": 1,
                "scope": "development validation only",
                "formal_test_used": False,
                "mask_seed": args.mask_seed,
                "masks_per_k": args.masks_per_k,
                "masks": masks,
            },
        )

    raw = normalize_raw(np.load(args.raw_dir / "val_data.npy", mmap_mode="r"))
    raw_labels = np.load(args.raw_dir / "val_labels.npy", mmap_mode="r")
    raw_paths = np.load(args.raw_dir / "val_path.npy", allow_pickle=True).astype(str)
    clean_labels = np.load(args.clean_qrs / "val_labels.npy", mmap_mode="r")
    clean_path_file = args.clean_qrs / "val_path.npy"
    clean_paths = (
        np.load(clean_path_file, allow_pickle=True).astype(str)
        if clean_path_file.is_file()
        else raw_paths
    )
    if not np.array_equal(raw_labels, clean_labels) or not np.array_equal(raw_paths, clean_paths):
        raise RuntimeError("raw and clean QRS validation labels/paths differ")
    truth = np.asarray(raw_labels, dtype=np.float32)
    stored = load_existing_standard(args.standard_deepsets_predictions, truth) if args.standard_deepsets_predictions else {"available": False}
    json_dump(
        args.output / "input-manifest.json",
        {
            "schema_version": 1,
            "scope": "development validation only",
            "formal_test_used": False,
            "raw_dir": str(args.raw_dir),
            "clean_qrs": str(args.clean_qrs),
            "raw_records": int(len(raw)),
            "labels_shape": list(truth.shape),
            "raw_labels_sha256": hashlib.sha256(np.asarray(raw_labels).tobytes()).hexdigest(),
            "raw_paths_sha256": hashlib.sha256(np.asarray(raw_paths).tobytes()).hexdigest(),
            "stored_standard_deepsets": {k: v for k, v in stored.items() if k != "score"},
        },
    )

    qrs_root = args.output / "qrs"
    qrs_root.mkdir(exist_ok=True)
    # k=12 is deliberately tokenized too, so the raw intervention and QRS
    # output are independently checked against the clean validation QRS.
    for count in args.counts:
        for mask_index, keep in enumerate(masks[str(count)]):
            variant = f"k{count}-mask{mask_index:02d}"
            make_qrs_variant(raw, np.asarray(raw_labels), raw_paths, qrs_root / variant, keep, variant)

    if 12 in args.counts:
        k12_dir = qrs_root / "k12-mask00"
        clean_signature = qrs_signature(args.clean_qrs)
        k12_signature = qrs_signature(k12_dir)
        qrs_diffs = {}
        for name in clean_signature:
            qrs_diffs[name] = {
                "same_sha256": clean_signature[name]["sha256"] == k12_signature[name]["sha256"],
                "clean_sha256": clean_signature[name]["sha256"],
                "k12_sha256": k12_signature[name]["sha256"],
            }
        alignment: dict[str, Any] = {
            "schema_version": 1,
            "scope": "development validation only",
            "formal_test_used": False,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "tolerance": args.tol,
            "qrs_signature_clean": clean_signature,
            "qrs_signature_k12": k12_signature,
            "qrs_file_comparison": qrs_diffs,
            "models": {},
        }
        # Compare each model's standard-forward logits on clean QRS and the
        # independently tokenized k=12 QRS.  This is the non-tautological gate.
        clean_hug = predict_fair(args.root, args.clean_qrs, "CLEAR_MASKED_HUG_finetune_base", args.hug_checkpoint, args.classes, args.seed, args.batch_size, args.num_workers)
        k12_hug = predict_fair(args.root, k12_dir, "CLEAR_MASKED_HUG_finetune_base", args.hug_checkpoint, args.classes, args.seed, args.batch_size, args.num_workers)
        clean_deep = predict_fair(args.root, args.clean_qrs, "CLEAR_MASKED_DEEPSETS_finetune_base", args.deepsets_checkpoint, args.classes, args.seed, args.batch_size, args.num_workers)
        k12_deep = predict_fair(args.root, k12_dir, "CLEAR_MASKED_DEEPSETS_finetune_base", args.deepsets_checkpoint, args.classes, args.seed, args.batch_size, args.num_workers)
        clean_hilar = predict_hilar(args.root, args.clean_qrs, args.deepsets_checkpoint, args.hilar_checkpoint, args.classes, args.seed, args.batch_size, args.num_workers)
        k12_hilar = predict_hilar(args.root, k12_dir, args.deepsets_checkpoint, args.hilar_checkpoint, args.classes, args.seed, args.batch_size, args.num_workers)
        for name, clean, candidate in (
            ("HUG", clean_hug, k12_hug),
            ("DeepSets", clean_deep, k12_deep),
            ("HiLAR / Final Direct", clean_hilar, k12_hilar),
        ):
            if not np.array_equal(clean["truth"], candidate["truth"]):
                raise RuntimeError(f"{name}: k=12 labels differ from clean validation")
            diff = np.abs(clean["logits"] - candidate["logits"])
            row = {
                "max_abs_logit_diff": float(diff.max()),
                "mean_abs_logit_diff": float(diff.mean()),
                "max_abs_probability_diff": float(
                    np.abs(
                        (1.0 / (1.0 + np.exp(-clean["logits"])))
                        - (1.0 / (1.0 + np.exp(-candidate["logits"])))
                    ).max()
                ),
                "clean_metrics": macro_metrics(clean["truth"], 1.0 / (1.0 + np.exp(-clean["logits"].astype(np.float64)))),
                "k12_metrics": macro_metrics(candidate["truth"], 1.0 / (1.0 + np.exp(-candidate["logits"].astype(np.float64)))),
            }
            alignment["models"][name] = row
        if stored.get("available"):
            standard_score = stored["score"]
            curve_score = 1.0 / (1.0 + np.exp(-clean_deep["logits"].astype(np.float64)))
            alignment["stored_standard_deepsets_max_abs_probability_diff"] = float(np.abs(standard_score - curve_score).max())
        alignment["gate_pass"] = bool(
            all(
                row["max_abs_logit_diff"] <= args.tol
                for row in alignment["models"].values()
            )
        )
        json_dump(args.output / "k12-alignment.json", alignment)
        json_dump(
            args.output / "k12-gate.json",
            {
                "schema_version": 1,
                "scope": "development validation only",
                "formal_test_used": False,
                "gate_pass": alignment["gate_pass"],
                "tolerance": args.tol,
                "max_abs_logit_diff": {
                    name: row["max_abs_logit_diff"] for name, row in alignment["models"].items()
                },
            },
        )
        if not alignment["gate_pass"]:
            raise SystemExit("k=12 alignment gate failed; refusing k=2..10")

    if any(count != 12 for count in args.counts):
        gate_path = args.output / "k12-gate.json"
        if not gate_path.is_file() or not json.loads(gate_path.read_text(encoding="utf-8")).get("gate_pass"):
            raise SystemExit("k=12 gate is absent or failed; refusing k=2..10")

    # Store predictions and metrics only after the k=12 gate has passed.
    metrics_rows: list[dict[str, Any]] = []
    for count in args.counts:
        for mask_index, keep in enumerate(masks[str(count)]):
            qrs_dir = qrs_root / f"k{count}-mask{mask_index:02d}"
            pred_root = args.output / "predictions" / f"k{count}" / f"mask{mask_index:02d}"
            pred_root.mkdir(parents=True, exist_ok=True)
            if count == 12:
                # Reuse the gated outputs for consistency and avoid another
                # stochastic forward pass.
                predictions = {
                    "HUG": predict_fair(args.root, qrs_dir, "CLEAR_MASKED_HUG_finetune_base", args.hug_checkpoint, args.classes, args.seed, args.batch_size, args.num_workers),
                    "DeepSets": predict_fair(args.root, qrs_dir, "CLEAR_MASKED_DEEPSETS_finetune_base", args.deepsets_checkpoint, args.classes, args.seed, args.batch_size, args.num_workers),
                    "HiLAR / Final Direct": predict_hilar(args.root, qrs_dir, args.deepsets_checkpoint, args.hilar_checkpoint, args.classes, args.seed, args.batch_size, args.num_workers),
                }
            else:
                predictions = {
                    "HUG": predict_fair(args.root, qrs_dir, "CLEAR_MASKED_HUG_finetune_base", args.hug_checkpoint, args.classes, args.seed, args.batch_size, args.num_workers),
                    "DeepSets": predict_fair(args.root, qrs_dir, "CLEAR_MASKED_DEEPSETS_finetune_base", args.deepsets_checkpoint, args.classes, args.seed, args.batch_size, args.num_workers),
                    "HiLAR / Final Direct": predict_hilar(args.root, qrs_dir, args.deepsets_checkpoint, args.hilar_checkpoint, args.classes, args.seed, args.batch_size, args.num_workers),
                }
            for model_name, prediction in predictions.items():
                filename = model_name.replace("/", "").replace(" ", "_") + ".npz"
                row = save_prediction(pred_root / filename, prediction, keep, args.seed, model_name)
                row.update({"count": count, "mask_index": mask_index, "qrs_dir": str(qrs_dir)})
                metrics_rows.append(row)
            json_dump(pred_root / "complete.json", {"count": count, "mask_index": mask_index, "mask": keep, "seed": args.seed, "models": list(predictions)})

    json_dump(
        args.output / "screen-summary.json",
        {
            "schema_version": 1,
            "scope": "development validation only",
            "formal_test_used": False,
            "seed": args.seed,
            "masks": masks,
            "metrics": metrics_rows,
        },
    )
    json_dump(
        args.output / "complete.json",
        {
            "schema_version": 1,
            "scope": "development validation only",
            "formal_test_used": False,
            "counts": args.counts,
            "seed": args.seed,
            "mask_seed": args.mask_seed,
            "k12_gate": str(args.output / "k12-gate.json"),
            "status": "complete",
        },
    )
    print(json.dumps({"status": "complete", "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

