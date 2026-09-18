#!/usr/bin/env python3
"""Extract one task's cache and train its validation-only HILA-K screen."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from extract_masked_features import extract
from protocol import TASKS
from train import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    extract(args.root.resolve(), args.task, device)
    train(args.root.resolve(), args.task, device)


if __name__ == "__main__":
    main()

