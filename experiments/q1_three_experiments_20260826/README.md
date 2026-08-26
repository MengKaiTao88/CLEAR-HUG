# Q1 three-experiment development campaign (2026-08-26)

This directory contains the reproducible, validation-only experiments requested
from the Q1 review discussion:

1. `set_transformer`: replace fixed lead grouping with a heartbeat-wise
   attention/Set-Transformer aggregation over the observed leads.
2. `anchor_ablation`: compare zero-initialized and randomly initialized local
   residual heads, with and without the logit anchor penalty.
3. `second_encoder`: validate the same lead-set aggregation with the frozen
   HeartLLM ECG encoder instead of CLEAR.  The intended paired comparison is
   `second-encoder-paired`, which keeps HeartLLM fixed and compares DeepSets
   with the HiLAR/Final Direct residual head.  The earlier `second-encoder`
   output is retained as an exploratory HeartLLM + attention-head run and
   must not be used as the DeepSets-versus-HiLAR result.

The campaign deliberately opens only `train` and `val` arrays.  It refuses
feature roots containing a `test` component and records `formal_test_used:
false` in every manifest.  It is not a replacement for the frozen six-task
formal table.

The server launcher scripts live beside this directory in each server's
`mvp/q1_three_experiments_20260826` directory.  Outputs are stored below
`results/q1-three-experiments-20260826` in the existing project root.
