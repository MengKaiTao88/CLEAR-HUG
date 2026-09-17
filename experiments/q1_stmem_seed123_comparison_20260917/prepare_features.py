#!/usr/bin/env python3
"""Extract frozen features from the audited custom ST-MEM seed-123 checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample


CAMPAIGN = "q1-stmem-seed123-six-task-three-fraction-seeds42-46-55-20260917"
SOURCE_CAMPAIGN = "q1-ecgfix-clocs-six-task-three-fraction-seeds42-46-55-20260916"
MODEL_NAME = "ST-MEM-seed123-custom-pretrain"
CHECKPOINT_SHA256 = "ad7e8527002213dda15e51c583108bf12aa01e1571935009e98b8e7f5ac09df9"
ARCHITECTURE_HASHES = {
    "st_mem_vit.py": "e1cfe021d429cd19d088f91746c03486759ac7c1d8fe16f57d294c32b44d0fdf",
    "vit.py": "4bb1954c6a7a2e83c301480895e2cfccb4ca69b4af0d519f0137d463364d9751",
}
BASES = ("ptbxl", "cpsc2018", "csn")
BASE_ARRAYS = {"ptbxl": "ptbxl_ecg_500hz.npy", "cpsc2018": "cpsc2018_ecg.npy",
               "csn": "csn_ecg.npy"}

# Import the already-audited task definitions and split/label/order audit.
_comparison = Path(__file__).resolve().parents[1] / "q1_ecgfounder_ecgjepa_comparison_20260917/prepare_features.py"
_spec = importlib.util.spec_from_file_location("founder_jepa_prepare", _comparison)
_module = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_module)
TASKS = _module.TASKS


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = path.with_suffix(path.suffix + ".incoming")
    incoming.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(incoming, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024): digest.update(block)
    return digest.hexdigest()


def audit_and_load(root: Path, device: torch.device):
    checkpoint = root / "model_weights/ST-MEM-seed123/pretrained_checkpoint.pt"
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("custom ST-MEM checkpoint SHA256 mismatch")
    encoder_dir = root / "external_models/ST-MEM-seed123/models/encoder"
    for name, expected in ARCHITECTURE_HASHES.items():
        if sha256(encoder_dir / name) != expected:
            raise RuntimeError(f"ST-MEM source identity mismatch: {name}")
    sys.path.insert(0, str(encoder_dir))
    from st_mem_vit import st_mem_vit_small
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    required = {"method": "ST_MEM", "seed": 123}
    if payload.get("method") != required["method"] or payload.get("seed") != required["seed"]:
        raise RuntimeError("checkpoint metadata does not identify ST_MEM seed 123")
    if (config.get("seq_len"), config.get("patch_size"), config.get("num_leads")) != (1000, 50, 12):
        raise RuntimeError("checkpoint input architecture metadata mismatch")
    state = {key[6:]: value for key, value in payload["model_state_dict"].items()
             if key.startswith("model.") and not key.startswith("model.head.")}
    model = st_mem_vit_small(num_leads=12, num_classes=None, seq_len=1000, patch_size=50)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"ST-MEM encoder load mismatch: {incompatible}")
    model = model.to(device).eval()
    for parameter in model.parameters(): parameter.requires_grad = False
    if sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):
        raise RuntimeError("ST-MEM encoder is not frozen")
    audit = {"model": MODEL_NAME, "checkpoint": str(checkpoint),
             "checkpoint_sha256": CHECKPOINT_SHA256, "checkpoint_seed": 123,
             "pretraining_protocol": payload.get("pretraining_protocol"),
             "architecture": "ST-MEM ViT-S/50", "input_shape": [12, 1000],
             "embedding_dim": 384, "encoder_frozen": True,
             "checkpoint_pretraining_scope": "historical unified PTB-XL five-class train split"}
    return model, audit


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base", choices=BASES, required=True); parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=32); args = parser.parse_args()
    root = args.root.resolve(); device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    model, audit = audit_and_load(root, device)
    task_audit = _module.audit_tasks(root)
    source = root / "results" / SOURCE_CAMPAIGN / "shared/raw"
    waves = np.load(source / BASE_ARRAYS[args.base], mmap_mode="r")
    campaign = root / "results" / CAMPAIGN
    output = campaign / "features" / MODEL_NAME / f"{args.base}.npy"
    done = output.with_suffix(".done.json")
    status = campaign / f"prepare-{args.base}-status.json"
    if done.is_file() and json.loads(done.read_text()).get("checkpoint_sha256") == CHECKPOINT_SHA256:
        atomic_json(status, json.loads(done.read_text())); return
    output.parent.mkdir(parents=True, exist_ok=True); incoming = output.with_suffix(".npy.incoming")
    features = np.lib.format.open_memmap(incoming, mode="w+", dtype=np.float32,
                                         shape=(len(waves), 384))
    with torch.inference_mode():
        for start in range(0, len(waves), args.batch_size):
            stop = min(start + args.batch_size, len(waves))
            # The checkpoint was trained on raw PTB-XL 100 Hz arrays without dataset normalization.
            values = resample(np.asarray(waves[start:stop], dtype=np.float32), 1000, axis=2).astype(np.float32)
            encoded = model.forward_encoding(torch.from_numpy(values).to(device, non_blocking=True))
            if encoded.shape != (stop - start, 384): raise RuntimeError(f"feature shape {encoded.shape}")
            features[start:stop] = encoded.float().cpu().numpy(); features.flush()
            atomic_json(status, {"state": "extracting", "base": args.base,
                "records_done": stop, "records_total": len(waves), "device": str(device), **audit})
    del features; os.replace(incoming, output)
    complete = {"state": "complete", "base": args.base, "records": len(waves),
                "device": str(device), "task_split_audit": task_audit, **audit}
    atomic_json(done, complete); atomic_json(status, complete)


if __name__ == "__main__": main()
