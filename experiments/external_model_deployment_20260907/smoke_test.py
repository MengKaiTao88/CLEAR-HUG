#!/usr/bin/env python3
"""GPU smoke tests for the pinned ST-MEM and HeartLang deployments."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint is not a mapping: {path}")
    return checkpoint


def test_st_mem(root: Path, device: torch.device) -> dict:
    repo = root / "external_models" / "ST-MEM"
    weight = root / "model_weights" / "ST-MEM" / "st_mem_vit_base_encoder.pth"
    sys.path.insert(0, str(repo))
    try:
        from models.encoder.st_mem_vit import st_mem_vit_base

        model = st_mem_vit_base(
            num_leads=12, num_classes=5, seq_len=2250, patch_size=75
        )
        state = load_checkpoint(weight)
        state = state.get("model", state.get("state_dict", state))
        current = model.state_dict()
        incompatible_shapes = [
            key
            for key, value in state.items()
            if key in current and tuple(value.shape) != tuple(current[key].shape)
        ]
        for key in incompatible_shapes:
            del state[key]
        missing, unexpected = model.load_state_dict(state, strict=False)
        model.eval().to(device)
        with torch.inference_mode():
            output = model(torch.randn(1, 12, 2250, device=device))
        torch.cuda.synchronize()
        return {
            "checkpoint": str(weight),
            "sha256": sha256(weight),
            "output_shape": list(output.shape),
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "removed_shape_mismatch_keys": incompatible_shapes,
            "passed": list(output.shape) == [1, 5],
        }
    finally:
        sys.path.remove(str(repo))
        for name in list(sys.modules):
            if name == "models" or name.startswith("models."):
                del sys.modules[name]


def test_heartlang(root: Path, device: torch.device) -> dict:
    repo = root / "external_models" / "HeartLang"
    pretrain_weight = root / "model_weights" / "HeartLang" / "checkpoint-200.pth"
    tokenizer_weight = (
        root / "model_weights" / "HeartLang" / "vqhbr-checkpoint-100.pth"
    )
    sys.path.insert(0, str(repo))
    try:
        import modeling_pretrain
        import modeling_vqhbr

        tokenizer = modeling_vqhbr.vqhbr()
        tok_state = load_checkpoint(tokenizer_weight)
        tok_state = tok_state.get("model", tok_state.get("state_dict", tok_state))
        for key in list(tok_state):
            if key.startswith(("loss", "teacher", "scaling")):
                del tok_state[key]
        tok_missing, tok_unexpected = tokenizer.load_state_dict(tok_state, strict=False)
        tokenizer.eval().to(device)
        waveform = torch.randn(1, 256, 96, device=device)
        with torch.inference_mode():
            _, token_ids, _ = tokenizer.encode(waveform)

        model = modeling_pretrain.HeartLang()
        pretrained = load_checkpoint(pretrain_weight)
        pretrained = pretrained.get("model", pretrained.get("state_dict", pretrained))
        missing, unexpected = model.load_state_dict(pretrained, strict=False)
        model.eval().to(device)
        token_ids = token_ids.reshape(1, -1).long()
        if token_ids.shape[1] != 256:
            raise RuntimeError(f"unexpected tokenizer sequence shape: {token_ids.shape}")
        # HeartLang consumes the beat waveform tensor; VQ-HBR token IDs are
        # masked-prediction targets, not its input embedding indices.
        mask = torch.zeros(token_ids.shape, dtype=torch.bool, device=device)
        channels = torch.arange(256, device=device).reshape(1, -1) % 12
        times = torch.arange(256, device=device).reshape(1, -1) % 16
        with torch.inference_mode():
            output = model(
                waveform,
                mask_bool_matrix=mask,
                in_chan_matrix=channels,
                in_time_matrix=times,
                return_all_tokens=True,
            )
        torch.cuda.synchronize()
        return {
            "pretrain_checkpoint": str(pretrain_weight),
            "pretrain_sha256": sha256(pretrain_weight),
            "tokenizer_checkpoint": str(tokenizer_weight),
            "tokenizer_sha256": sha256(tokenizer_weight),
            "tokenizer_missing_keys": list(tok_missing),
            "tokenizer_unexpected_keys": list(tok_unexpected),
            "pretrain_missing_keys": list(missing),
            "pretrain_unexpected_keys": list(unexpected),
            "token_shape": list(token_ids.shape),
            "output_shape": list(output.shape),
            "passed": list(output.shape) == [1, 257, 768],
        }
    finally:
        sys.path.remove(str(repo))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/107552503710"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "deployment_logs" / "smoke-test.json"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; GPU smoke test is required")
    device = torch.device("cuda:0")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "st_mem": test_st_mem(args.root, device),
        "heartlang": test_heartlang(args.root, device),
    }
    report["passed"] = report["st_mem"]["passed"] and report["heartlang"]["passed"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
