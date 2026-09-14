"""Pinned protocol for the official ST-MEM downstream reproduction."""

from protocol import NODES, NODE_TASKS, SEEDS, STMEM_SHA256, TASKS

CAMPAIGN = "q1-stmem-official-finetune-seeds42-46-55-20260914"
SOURCE_CAMPAIGN = "q1-heartlang-stmem-linear-probe-seeds42-46-55-20260914"
FRACTIONS = {"100pct": 1.0, "10pct": 0.10, "1pct": 0.01}
EPOCHS = 10
BATCH_SIZE = 16
BASE_LR = 1.0e-3
ACTUAL_LR = BASE_LR * BATCH_SIZE / 256
WARMUP_EPOCHS = 3
WEIGHT_DECAY = 0.05
SELECTION_METRIC = "validation_loss"


def specs(node: str):
    for fraction in FRACTIONS:
        for task in NODE_TASKS[node]:
            for seed in SEEDS:
                yield fraction, task, seed
