#!/usr/bin/env python3
"""Run the pinned ECG-FIX CLOCS six-task linear-probe campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


CAMPAIGN = "q1-ecgfix-clocs-six-task-three-fraction-seeds42-46-55-20260916"
ECGFIX_COMMIT = "991a31f14c94f72d5658bc172eae5736ba2fa149"
CLOCS_SHA256 = "039975cf563e76dd25a7975abfe1b74ff37308bd1e9fb24aaf6282f4ffdc5805"
DATASETS = ("PTBXL_form", "PTBXL_super", "PTBXL_sub", "PTBXL_rhythm", "CPSC", "CSN")
SEEDS = (42, 46, 55)
EXPECTED_UNITS_PER_SEED = len(DATASETS) * 3


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = path.with_suffix(path.suffix + ".incoming")
    incoming.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(incoming, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_deployment(root: Path, ecgfix: Path) -> dict:
    checkpoint = root / "model_weights/ecg-fix/best_weights_clocs"
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != CLOCS_SHA256:
        raise RuntimeError(f"CLOCS checkpoint SHA256 mismatch: {checkpoint_hash}")
    deployment_marker = ecgfix / ".deployment-complete"
    if not deployment_marker.is_file():
        raise RuntimeError("ECG-FIX deployment marker is missing")
    commit = deployment_marker.read_text(encoding="utf-8").strip()
    if commit != ECGFIX_COMMIT:
        raise RuntimeError(f"ECG-FIX commit mismatch: {commit}")
    return {
        "ecgfix_commit": commit,
        "clocs_checkpoint": str(checkpoint),
        "clocs_sha256": checkpoint_hash,
    }


def config(root: Path, seed: int) -> dict:
    campaign = root / "results" / CAMPAIGN
    shared = campaign / "shared"
    return {
        "raw_data_dir": str(shared / "raw"),
        "processed_dir": str(shared / "processed"),
        "embeddings_dir": str(shared / "embeddings"),
        "results_dir": str(campaign / "results" / f"seed-{seed}"),
        "tables_dir": str(campaign / "tables" / f"seed-{seed}"),
        "model_weights_dir": str(root / "model_weights/ecg-fix"),
        "physionet_dir": str(root / "data/ecg-fix-physionet"),
        "seed": seed,
        "dataset_roots": {
            "PTBXL": str(root / "data/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"),
            "CPSC": str(root / "data/ecg-fix-physionet/challenge-2020/1.0.2/training/cpsc_2018"),
            "CSN": str(root / "src/CLEAR-HUG/datasets/dataset_preprocess/CSN"),
            "ECHO_NEXT": str(root / "data/ecg-fix-physionet/echonext/1.1.0"),
        },
        "preprocess_workers": 30,
        "embedding": {"chunk_size": 5, "batch_size": 256, "num_workers": 0, "prefetch_factor": 0},
        "multi_process_eval": 4,
        "eval": {"batch_size": 256, "num_workers": 0, "sklearn_n_jobs": 1},
        "stats_tests": {"alpha": 0.01, "n_boot": 1000, "n_perm": 1000},
    }


def done_payloads(results_dir: Path, seed: int) -> list[dict]:
    payloads = []
    for path in sorted((results_dir / "done").glob("*_done.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("done") is True and value.get("seed") == seed:
            payloads.append(value)
    return payloads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "eval"), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    args = parser.parse_args()
    root = args.root.resolve()
    ecgfix = root / "external_models/ecg-fix"
    campaign = root / "results" / CAMPAIGN
    audit = verify_deployment(root, ecgfix)
    sys.path.insert(0, str(ecgfix))
    os.chdir(ecgfix)
    from src.eval import main_eval
    from src.metrics.export_results import main_export
    from src.preprocess.process_data import main_preprocess

    selected = SimpleNamespace(datasets=list(DATASETS), models=["CLOCS"])
    cfg = config(root, args.seed)
    if args.phase == "prepare":
        status = campaign / "prepare-status.json"
        atomic_json(status, {"state": "running", "datasets": DATASETS, "model": "CLOCS", **audit})
        main_preprocess(selected, cfg)
        marker = {
            "state": "complete", "datasets": DATASETS, "model": "CLOCS",
            "shared_embeddings_dir": cfg["embeddings_dir"], **audit,
        }
        atomic_json(campaign / "prepare-complete.json", marker)
        atomic_json(status, marker)
        return

    if not (campaign / "prepare-complete.json").exists():
        raise RuntimeError("shared preprocessing/embedding stage is not complete")
    status = campaign / f"seed-{args.seed}-status.json"
    atomic_json(status, {"state": "running", "seed": args.seed, "completed_units": 0,
                         "total_units": EXPECTED_UNITS_PER_SEED, **audit})
    main_eval(selected, cfg)
    completed = done_payloads(Path(cfg["results_dir"]), args.seed)
    if len(completed) != EXPECTED_UNITS_PER_SEED:
        failures = Path(cfg["results_dir"]) / "logs"
        raise RuntimeError(
            f"seed {args.seed}: expected {EXPECTED_UNITS_PER_SEED} valid done markers, "
            f"found {len(completed)}; inspect {failures}"
        )
    main_export(cfg)
    manifest = {
        "state": "complete", "seed": args.seed,
        "completed_units": len(completed), "total_units": EXPECTED_UNITS_PER_SEED,
        "datasets": DATASETS, "fractions": (0.01, 0.1, 1.0), "model": "CLOCS",
        "results_dir": cfg["results_dir"], "tables_dir": cfg["tables_dir"], **audit,
    }
    atomic_json(campaign / f"seed-{args.seed}-complete.json", manifest)
    atomic_json(status, manifest)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FATAL: {error!r}", file=sys.stderr, flush=True)
        raise
