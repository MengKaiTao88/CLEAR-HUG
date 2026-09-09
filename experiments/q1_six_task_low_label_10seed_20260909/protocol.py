"""Frozen protocol for the six-task 1%/10% ten-seed campaign."""

TASKS = {
    "superdiagnostic": (5, "PTBXL_QRS/superdiagnostic", 17084),
    "subdiagnostic": (23, "PTBXL_QRS/subdiagnostic", 17084),
    "form": (19, "PTBXL_QRS/form", 7197),
    "rhythm": (12, "PTBXL_QRS/rhythm", 16832),
    "cpsc2018": (9, "CPSC2018_QRS/data", 4950),
    "csn": (38, "CSN_QRS/data", 16546),
}
SEEDS = tuple(range(42, 52))
FRACTIONS = {"1pct": 0.01, "10pct": 0.10}
CAMPAIGN = "q1-six-task-low-label-10seed-20260909"
INPUT_CAMPAIGN = "q1-clear-deepsets-hilar-10seed-20260903"
RELEASED_SHA256 = "15e456964c5f819aa882522a203946b515cc1034e217dfcbc474e5e50bf1719a"

BASE_ASSIGNMENTS = {
    "10110": (
        "subdiagnostic:42", "cpsc2018:42", "csn:43", "csn:46", "csn:49",
        "subdiagnostic:45", "subdiagnostic:48", "subdiagnostic:51",
        "rhythm:44", "rhythm:47", "rhythm:50",
        "superdiagnostic:43", "superdiagnostic:46", "superdiagnostic:49",
        "form:44", "form:47", "form:50", "cpsc2018:45", "cpsc2018:48", "cpsc2018:51",
    ),
    "10092": (
        "csn:44", "csn:42", "csn:47", "csn:50", "csn:51",
        "subdiagnostic:43", "subdiagnostic:46", "subdiagnostic:49",
        "rhythm:42", "rhythm:45", "rhythm:48",
        "superdiagnostic:44", "superdiagnostic:47", "superdiagnostic:50",
        "form:45", "form:48", "form:51", "cpsc2018:43", "cpsc2018:46", "cpsc2018:49",
    ),
    "10103": (
        "rhythm:43", "superdiagnostic:42", "form:42", "csn:45", "csn:48",
        "subdiagnostic:44", "subdiagnostic:47", "subdiagnostic:50",
        "rhythm:46", "rhythm:49", "rhythm:51",
        "superdiagnostic:45", "superdiagnostic:48", "superdiagnostic:51",
        "form:43", "form:46", "form:49", "cpsc2018:44", "cpsc2018:47", "cpsc2018:50",
    ),
}

ASSIGNMENTS = {
    node: tuple(f"{fraction}:{spec}" for fraction in FRACTIONS for spec in specs)
    for node, specs in BASE_ASSIGNMENTS.items()
}

def validate():
    expected = {f"{fraction}:{task}:{seed}" for fraction in FRACTIONS for task in TASKS for seed in SEEDS}
    actual = [spec for specs in ASSIGNMENTS.values() for spec in specs]
    assert len(actual) == 120 and len(set(actual)) == 120 and set(actual) == expected
    assert all(len(specs) == 40 for specs in ASSIGNMENTS.values())

validate()
