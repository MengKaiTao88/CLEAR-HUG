#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from protocol import CAMPAIGN,EXPECTED_UNITS,FRACTIONS,SEEDS,TASKS

def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 expected={(f,t,s) for f in FRACTIONS for t in TASKS for s in SEEDS}; rows={}; hashes={'hila':set(),'hilar':set()}
 for path in a.results.rglob('train-val-complete.json'):
  row=json.loads(path.read_text()); key=(row.get('fraction_key'),row.get('task'),row.get('seed'))
  if key not in expected or key in rows: raise RuntimeError(f'unexpected/duplicate {key}: {path}')
  want=int(TASKS[key[1]][2]*FRACTIONS[key[0]])
  assert row['status']=='complete' and row['train_records']==want and row['selection_metric']=='macro_auroc'
  assert row['test_used_for_selection'] is False and row['feature_checkpoint_exact'] is True
  for name in hashes:
   value=row[f'{name}_sha256']; assert len(value)==64; hashes[name].add(value)
  rows[key]=row
 missing=expected-set(rows)
 if missing: raise RuntimeError(f'missing {len(missing)} manifests: {sorted(missing)[:5]}')
 assert all(len(v)==EXPECTED_UNITS for v in hashes.values()),{k:len(v) for k,v in hashes.items()}
 result={'schema_version':1,'status':'passed','campaign':CAMPAIGN,'train_val_units':EXPECTED_UNITS,'formal_test_authorized':True,'selection_metric':'macro_auroc','test_used_for_selection':False,'unique_checkpoint_hashes':{k:len(v) for k,v in hashes.items()}}
 if a.output.exists(): raise RuntimeError(f'refusing overwrite {a.output}')
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
