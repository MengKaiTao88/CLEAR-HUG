#!/usr/bin/env python3
"""Collect distributed manifests, build one gate, and deploy it identically."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

from common import atomic_json
from deploy_and_start import connect, mkdirs
from protocol import CAMPAIGN, FRACTIONS, MODELS, NODES, NODE_TASKS, SEEDS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collected = args.output / "collected-manifests"
    records = {}
    hashes = set()
    for node in NODES:
        client = connect(node)
        sftp = client.open_sftp()
        try:
            root = NODES[node][3]
            for task in NODE_TASKS[node]:
                for fraction in FRACTIONS:
                    for seed in SEEDS:
                        pair = {}
                        for model in MODELS:
                            relative = Path(fraction) / f"{task}-seed{seed}" / model / "training-complete.json"
                            remote = f"{root}/results/{CAMPAIGN}/{relative.as_posix()}"
                            with sftp.open(remote, "r") as handle:
                                data = handle.read()
                            local = collected / relative
                            local.parent.mkdir(parents=True, exist_ok=True)
                            incoming = local.with_name(local.name + ".incoming")
                            incoming.write_bytes(data); os.replace(incoming, local)
                            payload = json.loads(data)
                            if payload.get("status") != "complete" or payload.get("encoder_frozen") is not True:
                                raise RuntimeError(f"invalid training manifest {remote}")
                            if payload.get("selection_metric") != "validation_macro_auroc" or payload.get("test_used_for_selection") is not False:
                                raise RuntimeError(f"protocol violation {remote}")
                            checkpoint_hash = payload["checkpoint_sha256"]
                            if checkpoint_hash in hashes:
                                raise RuntimeError(f"duplicate checkpoint hash {checkpoint_hash}")
                            hashes.add(checkpoint_hash); pair[model] = payload
                        if pair["heartlang"]["subset_indices_sha256"] != pair["stmem"]["subset_indices_sha256"]:
                            raise RuntimeError(f"subset mismatch {fraction}:{task}:{seed}")
                        if pair["heartlang"]["initial_head_sha256"] != pair["stmem"]["initial_head_sha256"]:
                            raise RuntimeError(f"head initialization mismatch {fraction}:{task}:{seed}")
                        records[f"{fraction}:{task}:{seed}"] = {
                            model: pair[model]["checkpoint_sha256"] for model in MODELS
                        }
        finally:
            sftp.close(); client.close()
    if len(records) != 54 or len(hashes) != 108:
        raise RuntimeError(f"incomplete gate: records={len(records)} hashes={len(hashes)}")
    gate = {
        "status": "passed", "campaign": CAMPAIGN, "formal_test_authorized": True,
        "train_validation_units": 108, "task_seed_fraction_specs": 54,
        "selection_metric": "validation_macro_auroc", "test_used_for_selection": False,
        "paired_subset_and_head_initialization": True, "unique_checkpoint_hashes": 108,
        "records": records,
    }
    gate_path = args.output / "global-pretest-gate.json"
    atomic_json(gate_path, gate)
    expected = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    for node in NODES:
        client = connect(node); sftp = client.open_sftp()
        try:
            root = NODES[node][3]
            remote = f"{root}/results/{CAMPAIGN}/global-pretest-gate.json"
            mkdirs(sftp, str(PurePosixPath(remote).parent))
            incoming = remote + ".incoming"
            sftp.put(str(gate_path), incoming); sftp.chmod(incoming, 0o444); sftp.posix_rename(incoming, remote)
            with sftp.open(remote, "rb") as handle:
                actual = hashlib.sha256(handle.read()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"gate hash mismatch on {node}")
        finally:
            sftp.close(); client.close()
    print(json.dumps({"status": "passed", "manifests": 108, "gate_sha256": expected}, indent=2))


if __name__ == "__main__":
    main()
