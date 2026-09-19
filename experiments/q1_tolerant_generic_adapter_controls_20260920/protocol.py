"""Locked protocol for global-only parameter controls against TolerantECG + HiLAR."""

CAMPAIGN = "q1-tolerant-generic-adapter-controls-20260920"
FORMAL_TEST_CAMPAIGN = "q1-tolerant-generic-adapter-controls-formal-test-20260920"
LOW_LABEL_CAMPAIGN = "q1-tolerant-generic-adapter-controls-low-label-20260920"
REFERENCE_TEST_CAMPAIGN = "q1-tolerant-hilar-ecgfix-three-seed-formal-test-20260919"
BASELINE_CAMPAIGN = "q1-modern-mimic-baselines-six-task-three-fraction-seeds42-46-55-20260918"
HILAR_CAMPAIGN = "q1-tolerant-hilar-ecgfix-three-seed-20260919"
SEED42_HILAK_CAMPAIGN = "q1-tolerant-hilak-ecgfix-screen-seed42-20260918"
SEED42_LRA_CAMPAIGN = "q1-tolerant-hilak-lra-ecgfix-screen-seed42-20260919"

TASKS = ("PTBXL_form", "PTBXL_super", "PTBXL_sub", "PTBXL_rhythm", "CPSC", "CSN")
SEEDS = (42, 46, 55)
METHODS = ("parameter-matched-mlp", "generic-adapter")
FRACTION = 1.0
EMBED_DIM = 768
LEADS = 12
LEAD_EMBED_DIM = 32
HILA_HIDDEN_DIM = 1408
LRA_DIMS = (512, 256, 128)

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
MIN_EPOCHS = 10
PATIENCE = 12
GATE_INITIAL_VALUE = 0.1


def hilar_parameter_budget(classes: int) -> int:
    """Total trainable parameters introduced by locked HILA-K and LRA."""
    hila = (
        LEADS * LEAD_EMBED_DIM
        + (EMBED_DIM + LEAD_EMBED_DIM) * HILA_HIDDEN_DIM + HILA_HIDDEN_DIM
        + HILA_HIDDEN_DIM * HILA_HIDDEN_DIM + HILA_HIDDEN_DIM
        + HILA_HIDDEN_DIM * EMBED_DIM + EMBED_DIM
        + EMBED_DIM * classes + classes
        + classes
    )
    source = EMBED_DIM
    lra = 0
    for target in LRA_DIMS:
        lra += source * target + target
        source = target
    lra += source * classes + classes + classes
    return hila + lra
