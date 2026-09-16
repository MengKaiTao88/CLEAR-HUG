#!/usr/bin/env python3
"""Verify the pinned ECG-FIX CLOCS deployment on the 204 server."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ECGFIX_COMMIT = "991a31f14c94f72d5658bc172eae5736ba2fa149"
HF_REVISION = "6e2e3e6fb767df47e0b31e14ca3d73e1088c5134"
CLOCS_SHA256 = "039975cf563e76dd25a7975abfe1b74ff37308bd1e9fb24aaf6282f4ffdc5805"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/root/107552503710-1")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = Path(args.root)
    repo = root / "external_models" / "ecg-fix"
    weight = root / "model_weights" / "ecg-fix" / "best_weights_clocs"
    deployed_commit = (repo / ".deployment-complete").read_text().strip()
    if deployed_commit != ECGFIX_COMMIT:
        raise RuntimeError(f"ECG-FIX commit mismatch: {deployed_commit}")
    actual_hash = sha256(weight)
    if actual_hash != CLOCS_SHA256:
        raise RuntimeError(f"CLOCS checkpoint hash mismatch: {actual_hash}")

    sys.path.insert(0, str(repo))
    import torch
    from src.models.models import create_embedding_model

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = create_embedding_model("CLOCS", str(weight.parent)).to(device).eval()
    x = torch.zeros(2, 2500, 12, 2, device=device)
    with torch.no_grad():
        embedding = model(x)
    if tuple(embedding.shape) != (2, 256, 2):
        raise RuntimeError(f"unexpected embedding shape: {tuple(embedding.shape)}")

    print(json.dumps({
        "status": "ok",
        "ecgfix_commit": ECGFIX_COMMIT,
        "hf_revision": HF_REVISION,
        "checkpoint_sha256": actual_hash,
        "device": str(device),
        "input_shape": list(x.shape),
        "embedding_shape": list(embedding.shape),
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
    }, indent=2))


if __name__ == "__main__":
    main()
