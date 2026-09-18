#!/usr/bin/env python3
"""Prepare exact MERL-ordered embeddings and run one seed on one GPU."""
from __future__ import annotations
import argparse, fcntl
from pathlib import Path
from prepare_merl_embeddings import prepare_model
from protocol import CAMPAIGN, MODELS
from run_probe import main as probe_main


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True); parser.add_argument('--device',required=True)
    args=parser.parse_args(); root=args.root.resolve()
    lock_path=root/'results'/CAMPAIGN/'prepare.lock'; lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for model in MODELS:
            marker=root/'results'/CAMPAIGN/f'{model}-merl-embeddings-complete.json'
            if not marker.is_file(): prepare_model(root,model)
    import sys
    sys.argv=['run_probe.py','--root',str(root),'--seed',str(args.seed),'--device',args.device]
    probe_main()

if __name__=='__main__': main()
