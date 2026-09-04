"""Validate assignment coverage and campaign input arrays without mutation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from protocol import ASSIGNMENTS, CAMPAIGN, RELEASED_SHA256, SEEDS, TASKS, validate_protocol


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--node", choices=tuple(ASSIGNMENTS), required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Resolved campaign ecg_datasets directory (defaults to the original snapshot path).",
    )
    args = parser.parse_args()
    validate_protocol()
    if args.root.as_posix() != ASSIGNMENTS[args.node]["root"]:
        raise RuntimeError("node/root mismatch")
    checkpoint = args.root / "checkpoints/released_ckpt.pth"
    if sha256(checkpoint) != RELEASED_SHA256:
        raise RuntimeError("released checkpoint hash mismatch")
    inputs = args.input_root or (
        args.root / "campaign-inputs" / CAMPAIGN / "ecg_datasets"
    )
    arrays: dict[str, dict[str, object]] = {}
    for task, config in TASKS.items():
        dataset = inputs / str(config["dataset"])
        task_arrays = {}
        for split in ("train", "val", "test"):
            data_path = dataset / f"{split}_data.npy"
            labels_path = dataset / f"{split}_labels.npy"
            data = np.load(data_path, mmap_mode="r")
            labels = np.load(labels_path, mmap_mode="r")
            if len(data) != len(labels) or labels.shape[1] != config["classes"]:
                raise RuntimeError(f"{task}/{split} shape mismatch")
            if split == "test" and tuple(labels.shape) != tuple(config["test_shape"]):
                raise RuntimeError(f"{task} test shape mismatch: {labels.shape}")
            task_arrays[split] = {
                "records": len(labels), "classes": labels.shape[1],
                "data_sha256": sha256(data_path), "labels_sha256": sha256(labels_path),
            }
        arrays[task] = task_arrays
    payload = {
        "schema_version": 1, "status": "passed", "campaign": CAMPAIGN,
        "node": args.node, "root": str(args.root), "code_commit": args.commit,
        "input_root": str(inputs),
        "seeds": list(SEEDS), "specs": list(ASSIGNMENTS[args.node]["specs"]),
        "released_checkpoint_sha256": RELEASED_SHA256, "arrays": arrays,
    }
    output = args.root / "results" / CAMPAIGN / f"{args.node}-preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
