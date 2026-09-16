#!/usr/bin/env python3
"""Strictly verify the public Mehari--Strodthoff SimCLR/BYOL encoders."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

import torch


SOURCE_COMMIT = "956c6c17c9496b2ba459638c613c6efc148b95df"
WEIGHTS = {
    "simclr": "xresnet1d50_simclr_rrc_to_on_all.pt",
    "byol": "xresnet1d50_byol_rrc_to_on_all.pt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def unwrap(path: Path) -> dict[str, torch.Tensor]:
    # These legacy Lightning checkpoints contain pickled trainer metadata.  We
    # need tensor weights only and deliberately avoid importing/executing the
    # retired Lightning 1.1 runtime during deployment verification.
    # The archive also pickles one Lightning callback and the four upstream
    # transform objects.  Allowlist inert/imported types while keeping the
    # tensor-only unpickler; no training callback code is needed or executed.
    module = sys.modules.get("pytorch_lightning.callbacks.model_checkpoint")
    if module is None:
        package = sys.modules.setdefault("pytorch_lightning", types.ModuleType("pytorch_lightning"))
        callbacks = sys.modules.setdefault(
            "pytorch_lightning.callbacks", types.ModuleType("pytorch_lightning.callbacks")
        )
        module = types.ModuleType("pytorch_lightning.callbacks.model_checkpoint")
        sys.modules[module.__name__] = module
        package.callbacks = callbacks
        callbacks.model_checkpoint = module
    if not hasattr(module, "ModelCheckpoint"):
        model_checkpoint = type("ModelCheckpoint", (object,), {})
        model_checkpoint.__module__ = module.__name__
        module.ModelCheckpoint = model_checkpoint
    transform_module_name = "clinical_ts.timeseries_transformations"
    clinical_package = sys.modules.setdefault("clinical_ts", types.ModuleType("clinical_ts"))
    transform_module = sys.modules.get(transform_module_name)
    if transform_module is None:
        transform_module = types.ModuleType(transform_module_name)
        sys.modules[transform_module_name] = transform_module
        clinical_package.timeseries_transformations = transform_module
    transform_types = []
    for name in ("ToTensor", "TRandomResizedCrop", "TTimeOut", "Transpose"):
        if not hasattr(transform_module, name):
            value = type(name, (object,), {})
            value.__module__ = transform_module_name
            setattr(transform_module, name, value)
        transform_types.append(getattr(transform_module, name))
    torch.serialization.add_safe_globals(
        [module.ModelCheckpoint, *transform_types]
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: checkpoint is not a state dict")
    return payload


def encoder_state(method: str, checkpoint: dict[str, torch.Tensor], expected: dict) -> dict:
    mapped = {}
    for key in expected:
        suffix = key.removeprefix("features.")
        candidates = (
            key,
            f"encoder.features.{suffix}",
            f"online_network.encoder.features.{suffix}",
            f"model.features.{suffix}",
        )
        matches = [candidate for candidate in candidates if candidate in checkpoint]
        if len(matches) != 1:
            raise RuntimeError(f"{method}: {key} has source matches {matches}")
        value = checkpoint[matches[0]]
        if tuple(value.shape) != tuple(expected[key].shape):
            raise RuntimeError(
                f"{method}: {key} shape {tuple(value.shape)} != {tuple(expected[key].shape)}"
            )
        mapped[key] = value
    return mapped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    repository = root / "external_models/ecg-selfsupervised"
    weights = root / "model_weights/ecg-selfsupervised"
    sys.path.insert(0, str(repository))
    from models.resnet_simclr import ResNetSimCLR

    results = {}
    for method, filename in WEIGHTS.items():
        path = weights / filename
        model = ResNetSimCLR("xresnet1d50", out_dim=16, hidden=False)
        expected = model.features.state_dict()
        state = encoder_state(method, unwrap(path), expected)
        incompatible = model.features.load_state_dict(
            {key.removeprefix("features."): value for key, value in state.items()}, strict=True
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"{method}: incompatible state {incompatible}")
        model.features.eval()
        with torch.inference_mode():
            output = model.features(torch.randn(2, 12, 250))
        results[method] = {
            "checkpoint": str(path),
            "sha256": sha256(path),
            "checkpoint_tensor_count": len(unwrap(path)),
            "encoder_tensor_count": len(state),
            "encoder_parameters": sum(parameter.numel() for parameter in model.features.parameters()),
            "input_shape": [2, 12, 250],
            "output_shape": list(output.shape),
            "strict_encoder_load": True,
        }
    report = {
        "status": "ok",
        "source_repository": "https://github.com/tmehari/ecg-selfsupervised",
        "source_commit": SOURCE_COMMIT,
        "identity_note": (
            "These are the public Mehari--Strodthoff 2022 xresnet1d50 checkpoints; "
            "they are not asserted to be the independently trained MERL-paper baselines."
        ),
        "models": results,
    }
    destination = root / "deployment_logs/ecg-selfsupervised-simclr-byol-audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
