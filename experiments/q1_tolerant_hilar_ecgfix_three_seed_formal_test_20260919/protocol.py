"""Frozen formal-test protocol for the locked three-seed HiLAR campaign."""

CAMPAIGN = "q1-tolerant-hilar-ecgfix-three-seed-formal-test-20260919"
TRAIN_CAMPAIGN = "q1-tolerant-hilar-ecgfix-three-seed-20260919"
BASELINE_CAMPAIGN = "q1-modern-mimic-baselines-six-task-three-fraction-seeds42-46-55-20260918"
SEED42_HILAK_CAMPAIGN = "q1-tolerant-hilak-ecgfix-screen-seed42-20260918"
SEED42_LRA_CAMPAIGN = "q1-tolerant-hilak-lra-ecgfix-screen-seed42-20260919"

SEEDS = (42, 46, 55)
FRACTION = 1.0
SPLITS = ("test",)
TASKS = ("PTBXL_form", "PTBXL_super", "PTBXL_sub", "PTBXL_rhythm", "CPSC", "CSN")
EMBED_DIM = 768
LEADS = 12
LEAD_EMBED_DIM = 32
HILA_HIDDEN_DIM = 1408
LRA_DIMS = (512, 256, 128)
GATE_INITIAL_VALUE = 0.1
EXTRACTION_BATCH_SIZE = 8
SAMPLING_RATE = 500
MAX_BEATS = 24
REFRACTORY_SECONDS = 0.25
REFINE_SECONDS = 0.08
INTEGRATION_SECONDS = 0.12
FEATURE_WINDOW_RADIUS = 4
