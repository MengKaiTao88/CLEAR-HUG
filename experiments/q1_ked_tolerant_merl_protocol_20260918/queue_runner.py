#!/usr/bin/env python3
"""Launch three seed workers, one per idle GPU."""
from __future__ import annotations
import argparse, json, os, subprocess, time
from pathlib import Path
from protocol import CAMPAIGN, SEEDS


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); q=path.with_suffix(path.suffix+'.incoming')
    q.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n'); os.replace(q,path)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); a=p.parse_args(); root=a.root.resolve()
    here=Path(__file__).resolve().parent; out=root/'results'/CAMPAIGN; procs=[]
    for gpu,seed in enumerate(SEEDS):
        log=(out/f'seed-{seed}.log').open('ab',buffering=0)
        env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(gpu)
        proc=subprocess.Popen([str(root/'envs/ecg-fix/bin/python'),str(here/'worker.py'),'--root',str(root),
            '--seed',str(seed),'--device','cuda:0'],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        (out/f'seed-{seed}.pid').write_text(str(proc.pid)+'\n'); procs.append((seed,proc,log))
    atomic(out/'queue-status.json',{'state':'running','workers':{str(s):p.pid for s,p,_ in procs}})
    failed={}
    for seed,proc,log in procs:
        code=proc.wait(); log.close()
        if code: failed[str(seed)]=code
    atomic(out/'queue-status.json',{'state':'failed' if failed else 'complete','failed':failed})
    if failed: raise SystemExit(1)

if __name__=='__main__': main()
