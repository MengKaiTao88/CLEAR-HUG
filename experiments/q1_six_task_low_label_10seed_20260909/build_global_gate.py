#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from protocol import FRACTIONS, SEEDS, TASKS, CAMPAIGN

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifests',type=Path,required=True); p.add_argument('--preflights',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    expected={(f,t,s) for f in FRACTIONS for t in TASKS for s in SEEDS}; rows={}; hashes={m:set() for m in ('hug','hila','hilar')}
    for path in a.manifests.rglob('train-val-complete.json'):
        row=json.loads(path.read_text()); key=(row.get('fraction_key'),row.get('task'),row.get('seed'))
        if key not in expected or key in rows: raise RuntimeError(f'unexpected/duplicate {key}: {path}')
        _,_,total=TASKS[key[1]]; want=int(total*FRACTIONS[key[0]])
        assert row['status']=='complete' and row['train_records']==want and row['train_fraction']==FRACTIONS[key[0]]
        assert row['selection_metric']=='macro_auroc' and row['test_used_for_selection'] is False and row['feature_checkpoint_exact'] is True
        for m in hashes:
            value=row[f'{m}_sha256']; assert len(value)==64; hashes[m].add(value)
        rows[key]=row
    if set(rows)!=expected: raise RuntimeError(f'missing {len(expected-set(rows))} manifests')
    if any(len(v)!=120 for v in hashes.values()): raise RuntimeError({k:len(v) for k,v in hashes.items()})
    pre=[]
    for path in a.preflights.rglob('*-preflight.json'):
        row=json.loads(path.read_text());
        if row.get('status')=='passed' and row.get('node') in ('10110','10092','10103'): pre.append(row)
    by_node={r['node']:r for r in pre}; assert set(by_node)=={'10110','10092','10103'}
    reference=by_node['10110']['arrays']; assert all(r['arrays']==reference for r in by_node.values())
    result={'schema_version':1,'status':'passed','campaign':CAMPAIGN,'train_val_units':120,'model_checkpoints':360,'formal_test_authorized':True,'selection_metric':'macro_auroc','test_used_for_selection':False,'data_arrays_identical_across_nodes':True,'source_preflight_campaign':'q1-clear-deepsets-hilar-10seed-20260903','unique_checkpoint_hashes':{k:len(v) for k,v in hashes.items()},'runs':{f'{f}/{t}-seed{s}':rows[(f,t,s)] for f,t,s in sorted(expected)}}
    if a.output.exists(): raise RuntimeError(f'refusing overwrite {a.output}')
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))
if __name__=='__main__': main()
