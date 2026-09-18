"""Frozen protocol for the validation-only TolerantECG HILA-K screen."""

CAMPAIGN = "q1-tolerant-hilak-screen-seed42-20260918"
BASELINE_CAMPAIGN = "q1-ked-tolerant-merl-six-task-three-fraction-seeds42-46-55-20260918"
SOURCE_DATA_CAMPAIGN = "q1-ecgfix-clocs-six-task-three-fraction-seeds42-46-55-20260916"
SEED = 42
RATIO = 100
SPLITS = ("train", "val")
TASKS = ("superdiagnostic", "form", "cpsc2018", "csn")
EMBED_DIM = 768
LEADS = 12
LEAD_EMBED_DIM = 32
HIDDEN_DIM = 1408
BATCH_SIZE = 16
EXTRACTION_BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
MIN_EPOCHS = 10
PATIENCE = 12
WARMUP_EPOCHS = 5
GATE_INITIAL_VALUE = 0.1

