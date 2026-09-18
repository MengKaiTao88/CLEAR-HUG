"""Frozen protocol for the 2026-09-18 HiLAR rerun."""

CAMPAIGN = "q1-hilar-rerun-six-task-three-fraction-seeds42-46-55-20260918"
INPUT_ROOT_REL = "campaign-inputs/q1-clear-deepsets-hilar-10seed-20260903/ecg_datasets-resolved-v2"
TASKS = {
    "superdiagnostic": (5, "PTBXL_QRS/superdiagnostic", 17084),
    "subdiagnostic": (23, "PTBXL_QRS/subdiagnostic", 17084),
    "form": (19, "PTBXL_QRS/form", 7197),
    "rhythm": (12, "PTBXL_QRS/rhythm", 16832),
    "cpsc2018": (9, "CPSC2018_QRS/data", 4950),
    "csn": (38, "CSN_QRS/data", 16546),
}
FRACTIONS = {"1pct": 0.01, "10pct": 0.10, "100pct": 1.0}
SEEDS = (42, 46, 55)
EXPECTED_UNITS = len(TASKS) * len(FRACTIONS) * len(SEEDS)
RELEASED_SHA256 = "15e456964c5f819aa882522a203946b515cc1034e217dfcbc474e5e50bf1719a"

def specs_for_seed(seed: int):
    assert seed in SEEDS
    return tuple(f"{fraction}:{task}:{seed}" for fraction in FRACTIONS for task in TASKS)

def validate():
    specs = [x for seed in SEEDS for x in specs_for_seed(seed)]
    assert len(specs) == EXPECTED_UNITS == 54
    assert len(set(specs)) == EXPECTED_UNITS

validate()
