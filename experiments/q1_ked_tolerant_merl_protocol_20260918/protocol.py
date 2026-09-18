"""Frozen constants for the KED/TolerantECG MERL-split benchmark."""

CAMPAIGN = "q1-ked-tolerant-merl-six-task-three-fraction-seeds42-46-55-20260918"
SOURCE_CAMPAIGN = "q1-modern-mimic-baselines-six-task-three-fraction-seeds42-46-55-20260918"
SOURCE_EXPERIMENT = "q1_modern_mimic_baselines_20260918"
MODELS = ("KED", "TolerantECG")
SEEDS = (42, 46, 55)
RATIOS = (1, 10, 100)
SPLITS = ("train", "val", "test")
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
MIN_EPOCHS = 10
PATIENCE = 12
WARMUP_EPOCHS = 5

TASKS = {
    "superdiagnostic": {
        "source_dataset": "PTBXL_super", "kind": "ptbxl",
        "metadata": "ptbxl_diagnostic_class_metadata_final.csv",
        "split_dir": "ptbxl/super_class", "prefix": "ptbxl_super_class",
        "meta_columns": 6, "classes": 5,
        "counts": {"train": 17084, "val": 2146, "test": 2158},
    },
    "subdiagnostic": {
        "source_dataset": "PTBXL_sub", "kind": "ptbxl",
        "metadata": "ptbxl_diagnostic_subclass_metadata_final.csv",
        "split_dir": "ptbxl/sub_class", "prefix": "ptbxl_sub_class",
        "meta_columns": 6, "classes": 23,
        "counts": {"train": 17084, "val": 2146, "test": 2158},
    },
    "form": {
        "source_dataset": "PTBXL_form", "kind": "ptbxl",
        "metadata": "ptbxl_form_metadata_final.csv",
        "split_dir": "ptbxl/form", "prefix": "ptbxl_form",
        "meta_columns": 6, "classes": 19,
        "counts": {"train": 7197, "val": 901, "test": 880},
    },
    "rhythm": {
        "source_dataset": "PTBXL_rhythm", "kind": "ptbxl",
        "metadata": "ptbxl_rhythm_metadata_final.csv",
        "split_dir": "ptbxl/rhythm", "prefix": "ptbxl_rhythm",
        "meta_columns": 6, "classes": 12,
        "counts": {"train": 16832, "val": 2100, "test": 2098},
    },
    "cpsc2018": {
        "source_dataset": "CPSC", "kind": "cpsc",
        "metadata": "cpsc2018_metadata_final.csv",
        "split_dir": "icbeb", "prefix": "icbeb",
        "meta_columns": 7, "classes": 9,
        "counts": {"train": 4950, "val": 551, "test": 1376},
    },
    "csn": {
        "source_dataset": "CSN", "kind": "csn",
        "metadata": "csn_metadata_final.csv",
        "split_dir": "chapman", "prefix": "chapman",
        "meta_columns": 3, "classes": 38,
        "counts": {"train": 16546, "val": 1860, "test": 4620},
    },
}
