#!/usr/bin/env python3
"""Extract one model's embeddings and then run all frozen linear probes."""
from __future__ import annotations

import argparse
import subprocess
import sys
import traceback
from pathlib import Path

from prepare_embeddings import CAMPAIGN, MODELS, atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.root.resolve(); here = Path(__file__).resolve().parent
    campaign = root / "results" / CAMPAIGN
    status = campaign / f"{args.model}-status.json"
    try:
        subprocess.run([sys.executable, str(here / "prepare_embeddings.py"), "--root", str(root),
                        "--model", args.model, "--device", args.device], check=True)
        subprocess.run([sys.executable, str(here / "run_probe.py"), "--root", str(root),
                        "--model", args.model, "--device", args.device], check=True)
    except Exception as error:
        atomic_json(status, {"state": "failed", "model": args.model, "error": repr(error),
                             "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
