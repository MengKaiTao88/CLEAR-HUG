# Second-encoder paired result

This is the corrected encoder-transfer experiment.  The HeartLLM ECG encoder
is frozen and identical for both heads; only the heartbeat/lead aggregation
and local residual adapter differ.  The source QRS stream is reshaped from its
lead-major `(12 leads x 15 windows)` storage order into independent
`heartbeat x lead` tokens.  The old `second-encoder`/`heartllm-aligned`
directories are exploratory or use the invalid time-as-heartbeat mapping and
are not used for this comparison.

All values below are development validation results (PTB-XL
Superdiagnostic, folds 1--8 train and fold 9 validation).  The formal test
split was not opened.

| Seed | DeepSets AUROC | HiLAR AUROC | Δ AUROC (pp) | DeepSets AUPRC | HiLAR AUPRC | Δ AUPRC (pp) |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.80796 | 0.81484 | +0.688 | 0.57899 | 0.59202 | +1.302 |
| 43 | 0.80960 | 0.81468 | +0.507 | 0.58086 | 0.58695 | +0.609 |
| 44 | 0.80938 | 0.81263 | +0.325 | 0.58341 | 0.59091 | +0.751 |
| **Mean** | **0.80898** | **0.81405** | **+0.507** | **0.58109** | **0.58996** | **+0.887** |

The paired result is therefore **positive but modest**: HiLAR improves both
validation metrics for all three seeds, with mean gains of 0.507 AUROC points
and 0.887 AUPRC points.  This is evidence that the residual adapter can use
heartbeat/lead-local HeartLLM features under this frozen setup, but it is not a
strong claim of encoder-independent superiority.
