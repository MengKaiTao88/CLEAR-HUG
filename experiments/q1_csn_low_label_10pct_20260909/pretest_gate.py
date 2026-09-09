#!/usr/bin/env python3
import argparse, hashlib, json, os
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args(); seeds=(43,45,47); runs={}; hashes={'hug':set(),'hila':set(),'hilar':set()}
    for seed in seeds:
        run=args.root/f'csn-seed{seed}'
        payload=json.loads((run/'train-val-complete.json').read_text())
        assert payload['status']=='complete' and payload['seed']==seed
        assert payload['train_fraction']==0.1 and payload['train_records']==1654
        assert payload['selection_metric']=='macro_auroc' and payload['test_used_for_selection'] is False
        for key in ('hug','hila','hilar'):
            path=Path(payload[f'{key}_checkpoint']); actual=sha(path)
            assert actual==payload[f'{key}_sha256']; hashes[key].add(actual)
        runs[str(seed)]=payload
    assert all(len(value)==3 for value in hashes.values())
    result={'schema_version':1,'status':'passed','task':'csn','seeds':list(seeds),
      'train_fraction':0.1,'train_val_units':3,'model_checkpoints':9,
      'formal_test_authorized':True,'test_used_for_selection':False,'runs':runs}
    target=args.root/'pretest-gate.json'; incoming=target.with_suffix('.json.incoming')
    incoming.write_text(json.dumps(result,indent=2)+'\n'); os.replace(incoming,target)
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
