# Second-encoder paired result

This is the corrected encoder-transfer experiment.  The HeartLLM ECG encoder
is frozen and identical for both heads; only the lead aggregation and local
residual adapter differ.  The old `second-encoder` directory is an exploratory
HeartLLM + attention-head run and is not used for this comparison.

All values below are development validation results (PTB-XL
Superdiagnostic, folds 1--8 train and fold 9 validation).  The formal test
split was not opened.

| Seed | DeepSets AUROC | HiLAR AUROC | Δ AUROC (pp) | DeepSets AUPRC | HiLAR AUPRC | Δ AUPRC (pp) |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.79602 | 0.79644 | +0.042 | 0.56569 | 0.56690 | +0.121 |
| 43 | 0.80427 | 0.80446 | +0.019 | 0.57207 | 0.57230 | +0.023 |
| 44 | 0.79960 | 0.79974 | +0.014 | 0.56371 | 0.56370 | −0.001 |
| **Mean** | **0.79996** | **0.80021** | **+0.025** | **0.56716** | **0.56764** | **+0.048** |

The paired result is therefore **weakly positive but practically close to
neutral**: HiLAR improves the mean of both metrics, but the gain is much
smaller than the CLEAR-backbone result and is not uniformly positive for every
seed/metric.  The correct conclusion is that the residual adapter transfers to
the HeartLLM feature space only marginally under this frozen setup; it should
not be presented as a strong encoder-independent improvement.
