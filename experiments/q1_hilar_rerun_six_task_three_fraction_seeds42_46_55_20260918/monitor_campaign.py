#!/usr/bin/env python3
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args()
 train=list(a.results.rglob('train-val-complete.json')); formal=list(a.results.rglob('formal-test/complete.json'))
 statuses=[]
 for path in sorted(a.results.glob('gpu*-queue-status.json')):
  try: statuses.append(json.loads(path.read_text()))
  except Exception: pass
 out={'train_val_complete':len(train),'train_val_total':54,'formal_complete':len(formal),'formal_total':54,'queues':statuses}
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
