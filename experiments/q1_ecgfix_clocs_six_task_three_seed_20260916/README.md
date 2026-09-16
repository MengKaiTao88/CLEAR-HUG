# ECG-FIX CLOCS six-task, three-fraction, three-seed campaign

This campaign uses the pinned ECG-FIX repository and official `best_weights_clocs`
encoder to run linear probes for PTB-XL Form/Super/Sub/Rhythm, CPSC2018, and CSN.
Fractions are 1%, 10%, and 100%; seeds are 42, 46, and 55 (54 units total).

The queue waits until all three workers of
`q1-merl-five-task-three-seed-20260916` are complete and all 45 MERL completion
manifests exist. Shared preprocessing and CLOCS embeddings then run once on GPU
0. The three seed evaluations use separate result/table directories because the
upstream metric filenames do not include the seed.

Deploy and enqueue from the local repository with:

```bash
python experiments/q1_ecgfix_clocs_six_task_three_seed_20260916/deploy_queue.py
```

The remote campaign is stored entirely below `/root/107552503710-1/results`.
