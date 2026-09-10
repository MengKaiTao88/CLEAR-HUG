# Six-task 100%/10%/1% additional ten-seed campaign

This campaign trains CLEAR-HUG, HILA (masked DeepSets), and paired no-anchor
HiLAR from scratch for seeds 52--61 on six benchmark tasks at 100%, 10%, and
1% of the frozen training partition. Validation and test partitions remain
full and unchanged.

Each fraction/task/seed unit uses the same deterministic training subset for
all three models. Checkpoints are selected only by validation Macro AUROC.
Formal test remains closed until all 180 train/validation units pass the
global pre-test gate.

Remote result root:

`results/q1-six-task-three-fraction-seeds52-61-20260910`
