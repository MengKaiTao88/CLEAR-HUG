"""Verify a node's authoritative manifests against checkpoint bytes on disk."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from protocol import ASSIGNMENTS, CAMPAIGN


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
    args = parser.parse_args()
    assignment = ASSIGNMENTS[args.node]
    if args.root.as_posix() != assignment["root"]:
        raise RuntimeError("node/root mismatch")

    campaign = args.root / "results" / CAMPAIGN
    rows: dict[str, dict[str, object]] = {}
    hashes = {"hug": set(), "deepsets": set(), "hilar": set()}
    for spec in assignment["specs"]:
        task, seed_text = spec.split(":")
        run = campaign / f"{task}-seed{seed_text}"
        manifest_path = run / "train-val-complete.json"
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("task"), manifest.get("seed")) != (task, int(seed_text)):
            raise RuntimeError(f"manifest identity mismatch: {spec}")
        if manifest.get("feature_checkpoint_exact") is not True:
            raise RuntimeError(f"feature pairing not exact: {spec}")
        checked: dict[str, str] = {}
        for model in hashes:
            path = Path(str(manifest[f"{model}_checkpoint"]))
            expected = str(manifest[f"{model}_checkpoint_sha256"])
            actual = sha256(path)
            if actual != expected:
                raise RuntimeError(f"checkpoint hash mismatch: {spec}/{model}")
            hashes[model].add(actual)
            checked[model] = actual
        feature = json.loads(Path(str(manifest["feature_manifest"])).read_text())
        if Path(str(feature.get("checkpoint"))).resolve() != Path(
            str(manifest["deepsets_checkpoint"])
        ).resolve():
            raise RuntimeError(f"feature checkpoint path mismatch: {spec}")
        if feature.get("checkpoint_seed") != int(seed_text):
            raise RuntimeError(f"feature checkpoint seed mismatch: {spec}")
        rows[spec] = checked

    if any(len(values) != len(assignment["specs"]) for values in hashes.values()):
        raise RuntimeError("checkpoint hashes are not unique within node")
    payload = {
        "schema_version": 1,
        "status": "passed",
        "campaign": CAMPAIGN,
        "node": args.node,
        "units": len(rows),
        "actual_checkpoint_hashes": {k: len(v) for k, v in hashes.items()},
        "runs": rows,
    }
    output = campaign / f"{args.node}-checkpoint-audit.json"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
