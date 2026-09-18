#!/usr/bin/env python3
"""Run KED, D-BETA, and TolerantECG on one V100 each."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
import time
from pathlib import Path

import torch

from prepare_embeddings import CAMPAIGN, MODELS, atomic_json


def count_completed(campaign: Path) -> int:
    return sum(1 for path in campaign.glob("*/seed-*/*/*/complete.json")
               if json.loads(path.read_text(encoding="utf-8")).get("state") == "complete")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve()
    campaign = root / "results" / CAMPAIGN; campaign.mkdir(parents=True, exist_ok=True)
    status = campaign / "queue-status.json"
    if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
        atomic_json(status, {"state": "waiting_for_gpu", "required_gpus": 3,
                             "visible_gpus": torch.cuda.device_count()})
        raise RuntimeError("three visible CUDA GPUs are required")
    processes = []
    try:
        atomic_json(status, {"state": "running", "completed_units": count_completed(campaign),
                             "total_units": 162, "models": list(MODELS)})
        for gpu, model_name in enumerate(MODELS):
            if (campaign / f"{model_name}-complete.json").is_file():
                continue
            pidfile = campaign / f"{model_name}.pid"
            if pidfile.is_file():
                try:
                    os.kill(int(pidfile.read_text(encoding="utf-8").strip()), 0)
                    continue
                except (OSError, ValueError):
                    pass
            log = (campaign / f"{model_name}.log").open("a", encoding="utf-8")
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = [sys.executable, str(Path(__file__).with_name("worker.py")),
                       "--root", str(root), "--model", model_name, "--device", "cuda:0"]
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            pidfile.write_text(str(process.pid) + "\n", encoding="utf-8")
            processes.append((model_name, process, log))
        failures = []
        while True:
            completed_models = sum((campaign / f"{name}-complete.json").is_file() for name in MODELS)
            completed_units = count_completed(campaign)
            failed_models = []
            for name in MODELS:
                model_status = campaign / f"{name}-status.json"
                if model_status.is_file():
                    value = json.loads(model_status.read_text(encoding="utf-8"))
                    if value.get("state") == "failed":
                        failed_models.append({"model": name, "error": value.get("error")})
            atomic_json(status, {"state": "running" if not failed_models else "failed",
                                 "completed_models": completed_models,
                                 "completed_units": completed_units, "total_units": 162,
                                 "failed_models": failed_models})
            if failed_models or completed_models == len(MODELS):
                failures.extend(failed_models)
                break
            time.sleep(20)
        for model_name, process, log in processes:
            code = process.poll()
            if code not in (None, 0):
                failures.append({"model": model_name, "exit_code": code})
            log.close()
        completed_models = sum((campaign / f"{name}-complete.json").is_file() for name in MODELS)
        completed_units = count_completed(campaign)
        if failures or completed_models != len(MODELS) or completed_units != 162:
            raise RuntimeError(f"failures={failures}; models={completed_models}/3; units={completed_units}/162")
        value = {"state": "complete", "completed_models": completed_models,
                 "completed_units": completed_units, "total_units": 162}
        atomic_json(campaign / "campaign-complete.json", value); atomic_json(status, value)
    except Exception as error:
        atomic_json(status, {"state": "failed", "completed_units": count_completed(campaign),
                             "total_units": 162, "error": repr(error),
                             "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
