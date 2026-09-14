#!/usr/bin/env python3
"""Verify paired canary manifests before releasing full queues."""
from __future__ import annotations

import json

from deploy_and_start import connect
from protocol import CAMPAIGN, CANARIES, NODES


def main() -> None:
    report = {}
    for node, (_, fraction, task, seed) in CANARIES.items():
        client = connect(node)
        sftp = client.open_sftp()
        try:
            root = NODES[node][3]
            pair = {}
            for model in ("heartlang", "stmem"):
                path = f"{root}/results/{CAMPAIGN}/{fraction}/{task}-seed{seed}/{model}/training-complete.json"
                with sftp.open(path, "r") as handle:
                    pair[model] = json.loads(handle.read().decode())
            for model, payload in pair.items():
                if payload.get("status") != "complete" or payload.get("encoder_frozen") is not True:
                    raise RuntimeError(f"invalid {node} {model} canary")
                if payload.get("selection_metric") != "validation_macro_auroc" or payload.get("test_used_for_selection") is not False:
                    raise RuntimeError(f"protocol mismatch {node} {model} canary")
            if pair["heartlang"]["subset_indices_sha256"] != pair["stmem"]["subset_indices_sha256"]:
                raise RuntimeError(f"subset mismatch on {node}")
            if pair["heartlang"]["initial_head_sha256"] != pair["stmem"]["initial_head_sha256"]:
                raise RuntimeError(f"initial head mismatch on {node}")
            report[node] = {
                "status": "passed", "fraction": fraction, "task": task, "seed": seed,
                "subset_indices_sha256": pair["heartlang"]["subset_indices_sha256"],
                "initial_head_sha256": pair["heartlang"]["initial_head_sha256"],
                "validation_auroc": {model: pair[model]["best_validation_auroc"] for model in pair},
            }
        finally:
            sftp.close()
            client.close()
    print(json.dumps({"status": "passed", "canaries": report}, indent=2))


if __name__ == "__main__":
    main()
