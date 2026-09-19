# TolerantECG generic downstream adapter controls

This locked, validation-only campaign tests whether the gain of HILA-K + LRA can
be explained by downstream parameter count alone. It compares the existing
frozen-encoder LP and HiLAR results against two controls that consume **only the
global 768-D TolerantECG embedding**:

1. `parameter-matched-mlp`: a standalone `768 -> h -> h -> C` MLP head.
2. `generic-adapter`: a residual global adapter
   `z = z_LP + alpha * W_head(GELU(W_up(GELU(W_down(x)))))`, with frozen LP.

For every task, each control's trainable parameter count is matched within 1%
to the complete HILA-K + LRA trainable budget. Neither control receives
lead-masked features or heartbeat-local tokens. Seeds, splits, normalized
features, AdamW settings, batch size, early stopping, and validation Macro
AUROC selection are identical to the locked 100%-label HiLAR campaign. Test
data are not loaded or evaluated in this screening campaign.

Run on two GPUs:

```bash
nohup bash run_queue.sh 0 0 2 > gpu0.log 2>&1 &
nohup bash run_queue.sh 1 1 2 > gpu1.log 2>&1 &
```

After all 36 control runs complete:

```bash
/root/107552503710-1/.venv/bin/python summarize.py --root /root/107552503710-1
```

## Completed validation result

All 36 control runs completed without loading or evaluating test data. Mean
Macro AUROC over 6 tasks x 3 seeds was:

| Method | Mean validation AUROC | Gain vs LP | Wins vs LP |
|---|---:|---:|---:|
| TolerantECG + LP | 0.929514 | - | - |
| Parameter-matched MLP | 0.940271 | +1.076 pp | 14/18 |
| Generic adapter | 0.940198 | +1.068 pp | 17/18 |
| TolerantECG + HiLAR | **0.943950** | **+1.444 pp** | **18/18** |

HiLAR exceeded the parameter-matched MLP by 0.368 pp on average (14/18
pairs) and the generic adapter by 0.375 pp (15/18 pairs). Thus added global
downstream capacity explains part, but not all, of the HiLAR validation gain.
The machine-readable summary is stored at
`results/q1-tolerant-generic-adapter-controls-20260920/validation_summary.json`
on the experiment server.

## Frozen formal test

After validation selection was locked, the 36 generic-control checkpoints were
each evaluated on test exactly once. No architecture, parameter budget,
optimizer, or checkpoint criterion was changed.

| Method | Test AUROC | Test AUPRC | Macro-F1 | Exact ACC |
|---|---:|---:|---:|---:|
| TolerantECG + LP | 0.9206 | 0.5943 | 0.4881 | 0.5170 |
| Parameter-matched MLP | 0.9390 | 0.6131 | 0.5423 | 0.6292 |
| Generic adapter | 0.9378 | 0.6138 | 0.5429 | 0.6301 |
| TolerantECG + HiLAR | **0.9392** | **0.6196** | **0.5489** | **0.6355** |

The machine-readable formal-test summary is stored at
`results/q1-tolerant-generic-adapter-controls-formal-test-20260920/metrics_summary.json`
on the experiment server.

## Low-label validation controls

The locked global-only controls were also trained on the exact baseline subset
indices for 1% and 10% labels (72/72 runs, no test evaluation):

| Fraction | LP | Parameter-matched MLP | Generic adapter | HiLAR |
|---|---:|---:|---:|---:|
| 1% | 0.8089 | 0.8368 | 0.8392 | **0.8435** |
| 10% | 0.8872 | **0.9005** | 0.8915 | 0.8986 |

Values are mean validation Macro AUROC over 6 tasks x 3 seeds. HiLAR is best
at 1%; the parameter-matched MLP is best at 10%, so the low-label control does
not support an unconditional claim that HiLAR always beats generic capacity.
The machine-readable summary is stored at
`results/q1-tolerant-generic-adapter-controls-low-label-20260920/validation_summary.json`.
