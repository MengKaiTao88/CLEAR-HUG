# Six-task 1%/10% ten-seed campaign

This campaign trains CLEAR-HUG, HILA (masked DeepSets), and paired no-anchor
HiLAR from scratch for seeds 42--51 on six benchmark tasks at both 1% and 10%
of the frozen training partition. Validation and test partitions remain full.

Each fraction/task/seed unit uses the same deterministic random subset for all
three models. Checkpoints are selected only by validation Macro AUROC. Formal
test remains closed until all 120 train/validation units pass the global gate.

Remote result root:

`results/q1-six-task-low-label-10seed-20260909`
