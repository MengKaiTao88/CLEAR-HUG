"""Frozen first-protocol ECG-FIX HILA-K validation screen."""

CAMPAIGN = "q1-tolerant-hilak-ecgfix-screen-seed42-20260918"
BASELINE_CAMPAIGN = "q1-modern-mimic-baselines-six-task-three-fraction-seeds42-46-55-20260918"
SEED = 42
FRACTION = 1.0
SPLITS = ("train", "val")
TASKS = ("PTBXL_form", "PTBXL_super", "CPSC", "CSN")
EMBED_DIM = 768
LEADS = 12
LEAD_EMBED_DIM = 32
HIDDEN_DIM = 1408
BATCH_SIZE = 256
EXTRACTION_BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
MIN_EPOCHS = 10
PATIENCE = 12
GATE_INITIAL_VALUE = 0.1
