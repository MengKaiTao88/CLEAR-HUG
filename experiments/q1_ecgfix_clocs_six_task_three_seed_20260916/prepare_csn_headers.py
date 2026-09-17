#!/usr/bin/env python3
"""Restore WFDB headers for the Chapman/CSN MATLAB signal snapshot.

The CLEAR-HUG CSN snapshot contains the original ``val`` MATLAB signal files
and the audited train/validation/test CSVs, but not their small WFDB headers.
ECG-FIX intentionally reads CSN through WFDB, so this script reconstructs only
the missing headers.  Signals and multilabel targets are not rewritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat


SPLITS = ("chapman_train.csv", "chapman_val.csv", "chapman_test.csv")
META_COLUMNS = {"ecg_path", "age", "diagnose"}
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
EXPECTED_RECORDS = 23026
EXPECTED_LABELS = 38


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_code_map(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        mapping = {row["Acronym Name"].strip(): row["Snomed_CT"].strip() for row in rows}
    if not mapping:
        raise RuntimeError(f"empty SNOMED map: {path}")
    return mapping


def load_rows(base: Path) -> tuple[list[dict[str, str]], list[str], dict[str, str]]:
    all_rows: list[dict[str, str]] = []
    label_columns: list[str] | None = None
    split_hashes: dict[str, str] = {}
    seen: set[str] = set()
    for filename in SPLITS:
        path = base / filename
        split_hashes[filename] = sha256(path)
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            current_labels = [name for name in (reader.fieldnames or []) if name not in META_COLUMNS]
            if label_columns is None:
                label_columns = current_labels
            elif current_labels != label_columns:
                raise RuntimeError(f"label order differs in {filename}")
            for row in reader:
                record_id = Path(row["ecg_path"]).stem
                if record_id in seen:
                    raise RuntimeError(f"duplicate record across splits: {record_id}")
                seen.add(record_id)
                all_rows.append(row)
    if len(all_rows) != EXPECTED_RECORDS or len(seen) != EXPECTED_RECORDS:
        raise RuntimeError(f"expected {EXPECTED_RECORDS} unique rows, found {len(all_rows)}/{len(seen)}")
    if label_columns is None or len(label_columns) != EXPECTED_LABELS:
        raise RuntimeError(f"expected {EXPECTED_LABELS} labels, found {len(label_columns or [])}")
    return all_rows, label_columns, split_hashes


def signed_checksum(values: np.ndarray) -> int:
    value = int(np.asarray(values, dtype=np.int64).sum()) & 0xFFFF
    return value - 0x10000 if value >= 0x8000 else value


def header_text(record_id: str, signal: np.ndarray, dx_codes: list[str]) -> str:
    if signal.shape != (12, 5000) or signal.dtype.kind not in "iu":
        raise RuntimeError(f"unexpected signal for {record_id}: shape={signal.shape}, dtype={signal.dtype}")
    lines = [f"{record_id} 12 500 5000"]
    for lead_index, lead in enumerate(LEADS):
        channel = signal[lead_index]
        lines.append(
            f"{record_id}.mat 16+24 1000/mV 16 0 {int(channel[0])} "
            f"{signed_checksum(channel)} 0 {lead}"
        )
    lines.extend(("#Age: NaN", "#Sex: Unknown", f"#Dx: {','.join(dx_codes)}"))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    args = parser.parse_args()
    base = args.base_dir.resolve()
    records = base / "WFDBRecords"
    manifest_path = base / "ecgfix-csn-headers-complete.json"

    rows, labels, split_hashes = load_rows(base)
    code_map = load_code_map(base / "ConditionNames_SNOMED-CT.csv")
    missing_labels = sorted(set(labels) - set(code_map))
    if missing_labels:
        raise RuntimeError(f"labels missing from SNOMED map: {missing_labels}")

    written = 0
    for row in rows:
        relative = Path(row["ecg_path"].removeprefix("/chapman/").lstrip("/"))
        mat_path = base / relative
        if not mat_path.is_file():
            raise FileNotFoundError(mat_path)
        record_id = mat_path.stem
        header_path = mat_path.with_suffix(".hea")
        dx_codes = [code_map[label] for label in labels if int(row[label]) == 1]
        if header_path.exists():
            continue
        signal = np.asarray(loadmat(mat_path)["val"])
        incoming = header_path.with_suffix(".hea.incoming")
        incoming.write_text(header_text(record_id, signal, dx_codes), encoding="ascii")
        os.replace(incoming, header_path)
        written += 1

    mat_count = sum(1 for _ in records.rglob("*.mat"))
    header_count = sum(1 for _ in records.rglob("*.hea"))
    if mat_count != EXPECTED_RECORDS or header_count != EXPECTED_RECORDS:
        raise RuntimeError(f"CSN file count mismatch: mat={mat_count}, hea={header_count}")

    payload = {
        "state": "complete",
        "records": header_count,
        "labels": labels,
        "label_count": len(labels),
        "split_csv_sha256": split_hashes,
        "headers_written_this_run": written,
        "signal_files_modified": 0,
    }
    incoming = manifest_path.with_suffix(".json.incoming")
    incoming.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(incoming, manifest_path)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
