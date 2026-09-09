#!/usr/bin/env python3
import argparse, csv, json, os
from pathlib import Path
import numpy as np

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--train-fraction',type=float,default=0.1)
    args=parser.parse_args(); rows=[]
    for seed in (43,45,47):
        p=json.loads((args.root/f'csn-seed{seed}/formal-test/complete.json').read_text())
        row={'seed':seed}
        for model in ('hug','hila','hilar'):
            metrics=p[model]; row[f'{model}_macro_auroc']=metrics.get('macro_auroc',metrics.get('auroc')); row[f'{model}_macro_auprc']=metrics.get('macro_auprc',metrics.get('auprc'))
        rows.append(row)
    fields=list(rows[0]); csv_path=args.root/'seed-results.csv'
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary={'protocol':{'task':'csn','train_fraction':args.train_fraction,'seeds':[43,45,47],'test_used_for_selection':False},'models':{},'paired_deltas':{}}
    for model in ('hug','hila','hilar'):
        summary['models'][model]={}
        for metric in ('macro_auroc','macro_auprc'):
            values=np.array([r[f'{model}_{metric}'] for r in rows])
            summary['models'][model][metric]={'mean':float(values.mean()),'sample_sd':float(values.std(ddof=1)),'values':values.tolist()}
    for left,right in (('hila','hug'),('hilar','hila'),('hilar','hug')):
        name=f'{left}_minus_{right}'; summary['paired_deltas'][name]={}
        for metric in ('macro_auroc','macro_auprc'):
            delta=np.array([r[f'{left}_{metric}']-r[f'{right}_{metric}'] for r in rows])
            summary['paired_deltas'][name][metric]={'mean_pp':float(100*delta.mean()),'positive_seeds':int((delta>0).sum()),'values_pp':(100*delta).tolist()}
    target=args.root/'summary.json'; incoming=target.with_suffix('.json.incoming'); incoming.write_text(json.dumps(summary,indent=2)+'\n'); os.replace(incoming,target)
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
