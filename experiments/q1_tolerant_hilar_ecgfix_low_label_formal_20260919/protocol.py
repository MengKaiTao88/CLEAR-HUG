"""Locked 1%/10% TolerantECG + HiLAR validation and formal-test campaign."""

CAMPAIGN = "q1-tolerant-hilar-ecgfix-low-label-formal-20260919"
TRAIN_CAMPAIGN = CAMPAIGN
TEST_FEATURE_CAMPAIGN = "q1-tolerant-hilar-ecgfix-three-seed-formal-test-20260919"
BASELINE_CAMPAIGN = "q1-modern-mimic-baselines-six-task-three-fraction-seeds42-46-55-20260918"
LEAD_FEATURE_CAMPAIGN = "q1-tolerant-hilak-ecgfix-screen-seed42-20260918"
LOCAL_FEATURE_CAMPAIGN = "q1-tolerant-hilak-lra-ecgfix-screen-seed42-20260919"

SEEDS = (42, 46, 55)
FRACTIONS = (0.01, 0.1)
TASKS = ("PTBXL_form", "PTBXL_super", "PTBXL_sub", "PTBXL_rhythm", "CPSC", "CSN")
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
