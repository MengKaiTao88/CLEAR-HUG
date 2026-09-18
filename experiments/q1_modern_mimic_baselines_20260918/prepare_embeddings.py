#!/usr/bin/env python3
"""Extract frozen KED, D-BETA, or TolerantECG representations on ECG-FIX splits."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader


CAMPAIGN = "q1-modern-mimic-baselines-six-task-three-fraction-seeds42-46-55-20260918"
MODELS = ("KED", "D_BETA", "TolerantECG")
DATASETS = ("PTBXL_form", "PTBXL_super", "PTBXL_sub", "PTBXL_rhythm", "CPSC", "CSN")
SPLITS = ("train", "val", "test")
EMBED_DIM = 768
INITIAL_BATCH_SIZE = {"KED": 32, "D_BETA": 4, "TolerantECG": 128}
SOURCE_DATA_CAMPAIGN = "q1-ecgfix-clocs-six-task-three-fraction-seeds42-46-55-20260916"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = path.with_suffix(path.suffix + ".incoming")
    incoming.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(incoming, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def configure_ecgfix(root: Path):
    ecgfix = root / "external_models/ecg-fix"
    os.chdir(ecgfix)
    sys.path.insert(0, str(ecgfix))
    from src.models.models import create_embedding_model  # noqa: PLC0415
    from src.preprocess.datasets import make_dataset  # noqa: PLC0415
    from src.preprocess.CPSC import data_CPSC  # noqa: PLC0415
    from src.preprocess.CSN import data_CSN  # noqa: PLC0415
    from src.preprocess.PTBXL import data_PTBXL  # noqa: PLC0415
    shared = root / "results" / SOURCE_DATA_CAMPAIGN / "shared"
    cfg = {"raw_data_dir": str(shared / "raw"), "processed_dir": str(shared / "processed")}

    def current_config() -> dict:
        return cfg

    for module in (data_CPSC, data_CSN, data_PTBXL):
        module.load_config = current_config
    return ecgfix, create_embedding_model, make_dataset


def create_tolerant(experiment: Path, checkpoint: Path) -> torch.nn.Module:
    source = experiment / "vendor/tolerant_convnext.py"
    spec = importlib.util.spec_from_file_location("tolerant_convnext_official", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import official TolerantECG encoder from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.ConvNeXtV2(
        in_chans=12, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.0
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"TolerantECG state mismatch: missing={missing}, unexpected={unexpected}")
    return model


def model_and_provenance(root: Path, model_name: str):
    experiment = Path(__file__).resolve().parent
    weights = root / "model_weights/modern-mimic-baselines"
    ecgfix, create_embedding_model, make_dataset = configure_ecgfix(root)
    if model_name == "KED":
        checkpoint = weights / "ked.pt"
        model = create_embedding_model("KED", str(weights))
        source = "Zenodo record 14881564; ECG-FIX filename mirror"
    elif model_name == "D_BETA":
        checkpoint = weights / "dbeta_best.pt"
        config = weights / "dbeta_config.json"
        if not config.is_file():
            raise FileNotFoundError(config)
        model = create_embedding_model("D_BETA", str(weights))
        source = "doprakah/ecg-fix-weights mirror of released D-BETA checkpoint"
    elif model_name == "TolerantECG":
        checkpoint = weights / "TolerantECG_encoder.pth"
        model = create_tolerant(experiment, checkpoint)
        source = "ndhuynh02/TolerantECG official Hugging Face"
    else:
        raise ValueError(model_name)
    return model, make_dataset, {
        "model": model_name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_source": source,
        "ecgfix_path": str(ecgfix),
        "ecgfix_commit": subprocess.check_output(
            ["git", "-C", str(ecgfix), "rev-parse", "HEAD"], text=True
        ).strip(),
        "tolerant_source_commit": "f38ad823da11da4336d8a057cc5ff2c6f1693da8",
        "tolerant_encoder_source_sha256": sha256(experiment / "vendor/tolerant_convnext.py"),
    }


def get_features(model_name: str, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    if model_name == "D_BETA":
        ecg_model = model.ecg_encoder
        features, padding_mask = ecg_model.get_embeddings(x, padding_mask=None)
        cls = model.class_embedding.repeat((len(features), 1, 1))
        features = torch.cat([cls, features], dim=1)
        features = ecg_model.get_output(features, padding_mask)
        return model.unimodal_ecg_pooler(model.multi_modal_ecg_proj(features))
    if model_name == "KED":
        return model(x).mean(dim=2)
    return model(x)


def dataset_identity(ds) -> dict:
    indices = np.asarray(ds.indices, dtype=np.int64)
    labels = np.asarray(ds.labels[indices])
    return {
        "records": int(len(indices)),
        "source_indices_sha256": array_sha256(indices),
        "labels_sha256": array_sha256(labels),
        "label_shape": list(labels.shape),
    }


def extract_split(root: Path, model_name: str, model: torch.nn.Module, make_dataset,
                  dataset_name: str, split: str, device: torch.device, batch_size: int,
                  provenance: dict) -> int:
    output = root / "results" / CAMPAIGN / "shared/embeddings" / model_name / dataset_name / split
    done = output / "complete.json"
    if done.is_file():
        value = json.loads(done.read_text(encoding="utf-8"))
        if value.get("state") == "complete" and value.get("checkpoint_sha256") == provenance["checkpoint_sha256"]:
            return int(value["batch_size"])
    ds = make_dataset(dataset_name, split)
    identity = dataset_identity(ds)
    output.mkdir(parents=True, exist_ok=True)
    n = len(ds)
    _, y0 = ds[0]
    label_dim = int(np.asarray(y0).size)
    # All four PTB-XL tasks use the same ECG rows and split ordering. Encode the
    # waveform once and hard-link the immutable feature array after verifying
    # the exact source-index hash. Labels remain task-specific.
    if dataset_name.startswith("PTBXL_"):
        model_root = root / "results" / CAMPAIGN / "shared/embeddings" / model_name
        for candidate in DATASETS:
            if candidate == dataset_name or not candidate.startswith("PTBXL_"):
                continue
            source_dir = model_root / candidate / split
            source_done = source_dir / "complete.json"
            source_features = source_dir / "features.npy"
            if not source_done.is_file() or not source_features.is_file():
                continue
            source_meta = json.loads(source_done.read_text(encoding="utf-8"))
            if (source_meta.get("state") == "complete"
                    and source_meta.get("checkpoint_sha256") == provenance["checkpoint_sha256"]
                    and source_meta.get("source_indices_sha256") == identity["source_indices_sha256"]
                    and source_meta.get("feature_shape") == [n, EMBED_DIM]):
                incoming = output / "features.npy.incoming"
                incoming.unlink(missing_ok=True)
                try:
                    os.link(source_features, incoming)
                except OSError:
                    shutil.copyfile(source_features, incoming)
                labels = open_memmap(output / "labels.npy.incoming", mode="w+", dtype=np.float32,
                                     shape=(n, label_dim))
                labels[:] = np.asarray(ds.labels[np.asarray(ds.indices, dtype=np.int64)], dtype=np.float32)
                labels.flush(); del labels
                os.replace(incoming, output / "features.npy")
                os.replace(output / "labels.npy.incoming", output / "labels.npy")
                result = {
                    "state": "complete", "dataset": dataset_name, "split": split,
                    "batch_size": source_meta["batch_size"], "feature_shape": [n, EMBED_DIM],
                    "reused_features_from": candidate, **identity, **provenance,
                }
                atomic_json(done, result)
                return int(source_meta["batch_size"])
    while True:
        try:
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                                pin_memory=True, drop_last=False)
            embeddings = open_memmap(output / "features.npy.incoming", mode="w+", dtype=np.float32,
                                     shape=(n, EMBED_DIM))
            labels = open_memmap(output / "labels.npy.incoming", mode="w+", dtype=np.float32,
                                 shape=(n, label_dim))
            offset = 0
            model.eval()
            with torch.inference_mode():
                for x, y in loader:
                    x = x.to(device, non_blocking=True)
                    features = get_features(model_name, model, x)
                    if tuple(features.shape) != (len(x), EMBED_DIM):
                        raise RuntimeError(f"unexpected {model_name} feature shape {tuple(features.shape)}")
                    end = offset + len(x)
                    embeddings[offset:end] = features.float().cpu().numpy()
                    labels[offset:end] = y.float().numpy()
                    offset = end
            embeddings.flush(); labels.flush()
            del embeddings, labels, loader
            os.replace(output / "features.npy.incoming", output / "features.npy")
            os.replace(output / "labels.npy.incoming", output / "labels.npy")
            result = {
                "state": "complete", "dataset": dataset_name, "split": split,
                "batch_size": batch_size, "feature_shape": [n, EMBED_DIM],
                **identity, **provenance,
            }
            atomic_json(done, result)
            return batch_size
        except torch.cuda.OutOfMemoryError:
            for path in (output / "features.npy.incoming", output / "labels.npy.incoming"):
                path.unlink(missing_ok=True)
            gc.collect(); torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required for frozen embedding extraction")
    model, make_dataset, provenance = model_and_provenance(root, args.model)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != 0:
        raise RuntimeError(f"encoder is not frozen: {trainable} trainable parameters")
    campaign = root / "results" / CAMPAIGN
    status = campaign / f"{args.model}-status.json"
    batch_size = INITIAL_BATCH_SIZE[args.model]
    completed = 0
    for dataset_name in DATASETS:
        for split in SPLITS:
            atomic_json(status, {"state": "extracting", "model": args.model,
                                 "dataset": dataset_name, "split": split,
                                 "completed_embedding_splits": completed,
                                 "total_embedding_splits": len(DATASETS) * len(SPLITS),
                                 "batch_size": batch_size, **provenance})
            batch_size = extract_split(root, args.model, model, make_dataset, dataset_name,
                                       split, device, batch_size, provenance)
            completed += 1
    atomic_json(campaign / f"{args.model}-embeddings-complete.json", {
        "state": "complete", "model": args.model, "completed_embedding_splits": completed,
        "total_embedding_splits": len(DATASETS) * len(SPLITS), "batch_size": batch_size,
        **provenance,
    })


if __name__ == "__main__":
    main()
