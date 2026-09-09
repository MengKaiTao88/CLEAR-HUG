#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
import numpy as np
from protocol import FRACTIONS,SEEDS,TASKS

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args(); rows=[]; summary={}
 for f in FRACTIONS:
  summary[f]={}
  for task in TASKS:
   task_rows=[]
   for seed in SEEDS:
    x=json.loads((a.root/f/task/f'{task}-seed{seed}'/'formal-test/complete.json').read_text()); row={'fraction':f,'task':task,'seed':seed}
    for m in ('hug','hila','hilar'):
     row[f'{m}_auroc']=x[m].get('macro_auroc',x[m].get('auroc'));row[f'{m}_auprc']=x[m].get('macro_auprc',x[m].get('auprc'))
    rows.append(row);task_rows.append(row)
   out={'models':{},'paired_deltas':{}}
   for m in ('hug','hila','hilar'):
    out['models'][m]={k:{'mean':float(np.mean([r[f'{m}_{k}'] for r in task_rows])),'sample_sd':float(np.std([r[f'{m}_{k}'] for r in task_rows],ddof=1))} for k in ('auroc','auprc')}
   for l,r in (('hila','hug'),('hilar','hila'),('hilar','hug')):
    out['paired_deltas'][f'{l}_minus_{r}']={}
    for k in ('auroc','auprc'):
     d=np.asarray([x[f'{l}_{k}']-x[f'{r}_{k}'] for x in task_rows]);out['paired_deltas'][f'{l}_minus_{r}'][k]={'mean_pp':float(100*d.mean()),'positive_seeds':int((d>0).sum()),'values_pp':(100*d).tolist()}
   summary[f][task]=out
 fields=list(rows[0]);
 with (a.root/'seed-results.csv').open('w',newline='') as h: w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
 (a.root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
