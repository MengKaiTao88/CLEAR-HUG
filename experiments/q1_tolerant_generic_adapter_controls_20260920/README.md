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

### Low-label frozen formal test

All 72 frozen low-label control checkpoints were evaluated on test exactly
once. The resulting overall metrics are:

| Fraction | Method | AUROC | AUPRC | Macro-F1 | Exact ACC |
|---|---|---:|---:|---:|---:|
| 1% | LP | 0.7939 | **0.4301** | 0.2862 | 0.0629 |
| 1% | Parameter-matched MLP | 0.8336 | 0.4072 | **0.3429** | **0.5130** |
| 1% | Generic adapter | 0.8331 | 0.4108 | 0.3307 | 0.4456 |
| 1% | HiLAR | **0.8381** | 0.4277 | 0.3158 | 0.3923 |
| 10% | LP | 0.8758 | 0.5329 | 0.3525 | 0.1787 |
| 10% | Parameter-matched MLP | **0.8995** | **0.5402** | **0.4788** | **0.6015** |
| 10% | Generic adapter | 0.8891 | 0.5201 | 0.4372 | 0.5302 |
| 10% | HiLAR | 0.8958 | 0.5254 | 0.4020 | 0.5054 |

The machine-readable summary is stored at
`results/q1-tolerant-generic-adapter-controls-low-label-formal-test-20260920/metrics_summary.json`.
