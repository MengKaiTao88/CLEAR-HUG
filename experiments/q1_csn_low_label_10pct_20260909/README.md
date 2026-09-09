# CSN 10% labeled-data comparison

This campaign compares CLEAR-HUG, HILA (masked DeepSets), and no-anchor HiLAR
on CSN with seeds 43, 45, and 47. Each seed uses the same deterministic random
10% subset of the original CSN training partition. Validation and test remain
the complete frozen benchmark partitions.

Checkpoints are selected only by validation Macro AUROC. HiLAR is trained from
the matching frozen HILA checkpoint, logits, and local features. Formal test is
opened only after all nine train/validation checkpoints pass the pre-test gate.

Remote campaign root:

`/root/107552503710/results/q1-csn-low-label-10pct-20260909`

