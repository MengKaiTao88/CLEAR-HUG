#!/usr/bin/env python3
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args()
 grouped=defaultdict(list)
 for path in a.results.rglob('formal-test/complete.json'):
  row=json.loads(path.read_text());grouped[(row['fraction_key'],row['task'])].append(row)
 rows=[]
 for (fraction,task),vals in sorted(grouped.items()):
  for method in ('hila','hilar'):
   auroc=np.array([x[method]['auroc'] for x in vals]);auprc=np.array([x[method]['auprc'] for x in vals])
   rows.append({'fraction':fraction,'task':task,'method':method,'n':len(vals),'macro_auroc_mean':float(auroc.mean()),'macro_auroc_std':float(auroc.std(ddof=1)) if len(vals)>1 else None,'macro_auprc_mean':float(auprc.mean()),'macro_auprc_std':float(auprc.std(ddof=1)) if len(vals)>1 else None})
 payload={'formal_units':sum(len(x) for x in grouped.values()),'rows':rows}
 text=json.dumps(payload,indent=2)+'\n'; print(text,end='')
 if a.output:a.output.write_text(text)
if __name__=='__main__':main()
