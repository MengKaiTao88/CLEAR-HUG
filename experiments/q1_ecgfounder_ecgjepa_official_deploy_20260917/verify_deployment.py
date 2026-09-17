#!/usr/bin/env python3
"""Verify the pinned official ECGFounder and ECG-JEPA deployments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch


FOUNDER_COMMIT = "04edac702b61c91face519774ddcc0cd712fef23"
FOUNDER_HF_REVISION = "d9b1793951b2342f5f7e84f1ac03cd37f8a08724"
FOUNDER_SHA256 = "ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997"
JEPA_COMMIT = "d937ad2c2c8a1e22856ce7e4a23a30f84a71217c"
JEPA_DRIVE_ID = "1gMOT4xjQQg0GZkY1iE6NuDzua4ALw00l"
JEPA_SHA256 = "61334869f905a7d6de32bc573c60024eaf6efba7c35c0e45fc2ea7d52b6ff66e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = path.with_suffix(path.suffix + ".incoming")
    incoming.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(incoming, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    founder_source = root / "external_models/ECGFounder"
    founder_weight = root / "model_weights/ECGFounder/12_lead_ECGFounder.pth"
    jepa_source = root / "external_models/ECG_JEPA"
    jepa_weight = root / "model_weights/ECG_JEPA/multiblock_epoch100.pth"

    if (founder_source / ".deployment-complete").read_text().strip() != FOUNDER_COMMIT:
        raise RuntimeError("ECGFounder source commit marker mismatch")
    if (jepa_source / ".deployment-complete").read_text().strip() != JEPA_COMMIT:
        raise RuntimeError("ECG-JEPA source commit marker mismatch")
    founder_hash = sha256(founder_weight)
    jepa_hash = sha256(jepa_weight)
    if founder_hash != FOUNDER_SHA256:
        raise RuntimeError(f"ECGFounder checkpoint SHA256 mismatch: {founder_hash}")
    if jepa_hash != JEPA_SHA256:
        raise RuntimeError(f"ECG-JEPA checkpoint SHA256 mismatch: {jepa_hash}")

    sys.path.insert(0, str(founder_source))
    from finetune_model import ft_12lead_ECGFounder

    founder = ft_12lead_ECGFounder(device, founder_weight, n_classes=1, linear_prob=True)
    founder.eval()
    founder_trainable = {name: value.numel() for name, value in founder.named_parameters()
                         if value.requires_grad}
    with torch.no_grad():
        founder_output = founder(torch.zeros(1, 12, 5000, device=device))

    sys.path[0] = str(jepa_source)
    for name in ("models", "ecg_jepa", "pos_encoding"):
        sys.modules.pop(name, None)
    from models import load_encoder

    jepa, embed_dim = load_encoder(jepa_weight)
    jepa = jepa.to(device).eval()
    for value in jepa.parameters():
        value.requires_grad = False
    with torch.no_grad():
        jepa_output = jepa.representation(torch.zeros(1, 8, 2500, device=device))

    payload = {
        "state": "complete",
        "device": str(device),
        "ecgfounder": {
            "source": "https://github.com/PKUDigitalHealth/ECGFounder",
            "source_commit": FOUNDER_COMMIT,
            "checkpoint_source": "https://huggingface.co/PKUDigitalHealth/ECGFounder",
            "checkpoint_revision": FOUNDER_HF_REVISION,
            "checkpoint": str(founder_weight),
            "checkpoint_bytes": founder_weight.stat().st_size,
            "checkpoint_sha256": founder_hash,
            "input": [1, 12, 5000],
            "sampling_rate_hz": 500,
            "output": list(founder_output.shape),
            "linear_probe_trainable": founder_trainable,
        },
        "ecg_jepa": {
            "source": "https://github.com/sehunfromdaegu/ECG_JEPA",
            "source_commit": JEPA_COMMIT,
            "checkpoint_source": f"https://drive.google.com/file/d/{JEPA_DRIVE_ID}/view",
            "checkpoint_variant": "multi-block epoch 100",
            "checkpoint": str(jepa_weight),
            "checkpoint_bytes": jepa_weight.stat().st_size,
            "checkpoint_sha256": jepa_hash,
            "input": [1, 8, 2500],
            "sampling_rate_hz": 250,
            "representation": list(jepa_output.shape),
            "embedding_dim": embed_dim,
            "encoder_parameters": sum(value.numel() for value in jepa.parameters()),
            "encoder_trainable_parameters": sum(value.numel() for value in jepa.parameters()
                                                if value.requires_grad),
        },
    }
    output = args.output or root / "results/q1-ecgfounder-ecgjepa-official-deploy-20260917/deployment-manifest.json"
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
