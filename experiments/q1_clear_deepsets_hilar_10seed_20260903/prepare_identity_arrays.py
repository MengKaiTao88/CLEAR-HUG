"""Create audited patient-cluster identities for the formal-test bootstrap."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from protocol import TASKS


def truth(campaign: Path, task: str) -> np.ndarray:
    path = campaign / f"{task}-seed42/formal-test/paired/test_predictions.npz"
    with np.load(path, allow_pickle=False) as payload:
        return np.asarray(payload["y_true"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, required=True)
    parser.add_argument("--ptbxl-database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")

    metadata = pd.read_csv(args.ptbxl_database)
    patient_by_ecg = dict(zip(metadata["ecg_id"].astype(int), metadata["patient_id"]))
    ptb_paths = {
        "superdiagnostic": args.path_root / "PTBXL_QRS/subdiagnostic/test_path.npy",
        "subdiagnostic": args.path_root / "PTBXL_QRS/subdiagnostic/test_path.npy",
        "form": args.path_root / "PTBXL_QRS/form/test_path.npy",
        "rhythm": args.path_root / "PTBXL_QRS/rhythm/test_path.npy",
    }
    other_paths = {
        "cpsc2018": args.path_root / "CPSC2018_QRS/data/test_path.npy",
        "csn": args.path_root / "CSN_QRS/data/test_path.npy",
    }
    args.output.mkdir(parents=True)
    for task, config in TASKS.items():
        y = truth(args.campaign, task)
        source = ptb_paths.get(task, other_paths.get(task))
        paths = np.load(source, allow_pickle=True).astype(str)
        if len(paths) != len(y) or y.shape != tuple(config["test_shape"]):
            raise RuntimeError(f"identity/order shape mismatch: {task}")
        if task in ptb_paths:
            ecg_ids = []
            for value in paths:
                match = re.search(r"(\d+)_lr$", value)
                if match is None:
                    raise RuntimeError(f"cannot parse PTB-XL ECG id: {value}")
                ecg_ids.append(int(match.group(1)))
            patient_id = np.asarray([patient_by_ecg[item] for item in ecg_ids])
            source_kind = "PTB-XL patient_id mapped from official ptbxl_database.csv"
        else:
            patient_id = np.asarray([Path(value).stem for value in paths])
            if len(np.unique(patient_id)) != len(patient_id):
                raise RuntimeError(f"non-unique record identities require patient metadata: {task}")
            source_kind = "benchmark record id; one record per included subject"
        target = args.output / task
        target.mkdir()
        np.savez_compressed(target / "identity.npz", y_true=y, patient_id=patient_id)
        (target / "source.txt").write_text(source_kind + "\n")
        print(task, len(y), len(np.unique(patient_id)), source_kind)


if __name__ == "__main__":
    main()
