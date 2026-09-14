#!/usr/bin/env python3
"""Download and verify every distributed formal-test artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from deploy_and_start import connect
from protocol import CAMPAIGN, FRACTIONS, MODELS, NODES, NODE_TASKS, SEEDS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = 0
    for node in NODES:
        client = connect(node); sftp = client.open_sftp()
        try:
            root = NODES[node][3]
            for task in NODE_TASKS[node]:
                for fraction in FRACTIONS:
                    for seed in SEEDS:
                        for model in MODELS:
                            relative = Path(fraction) / f"{task}-seed{seed}" / model / "formal-test"
                            local = args.output / relative
                            local.mkdir(parents=True, exist_ok=True)
                            for filename in ("formal-test-result.json", "test_predictions.npz"):
                                target = local / filename
                                if not target.exists():
                                    incoming = target.with_name(filename + ".incoming")
                                    remote = f"{root}/results/{CAMPAIGN}/{relative.as_posix()}/{filename}"
                                    sftp.get(remote, str(incoming)); os.replace(incoming, target)
                            result = json.loads((local / "formal-test-result.json").read_text())
                            if result.get("status") != "complete" or result.get("test_used_for_selection") is not False:
                                raise RuntimeError(f"invalid formal result {local}")
                            if sha256(local / "test_predictions.npz") != result["predictions_sha256"]:
                                raise RuntimeError(f"prediction hash mismatch {local}")
                            count += 1
        finally:
            sftp.close(); client.close()
    print(json.dumps({"status": "complete", "formal_results": count}, indent=2))


if __name__ == "__main__":
    main()
