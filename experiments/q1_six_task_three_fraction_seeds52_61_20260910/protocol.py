"""Frozen protocol for the additional seeds 52--61 campaign."""

TASKS = {
    "superdiagnostic": (5, "PTBXL_QRS/superdiagnostic", 17084),
    "subdiagnostic": (23, "PTBXL_QRS/subdiagnostic", 17084),
    "form": (19, "PTBXL_QRS/form", 7197),
    "rhythm": (12, "PTBXL_QRS/rhythm", 16832),
    "cpsc2018": (9, "CPSC2018_QRS/data", 4950),
    "csn": (38, "CSN_QRS/data", 16546),
}
SEEDS = tuple(range(52, 62))
FRACTIONS = {"100pct": 1.0, "10pct": 0.10, "1pct": 0.01}
CAMPAIGN = "q1-six-task-three-fraction-seeds52-61-20260910"
INPUT_CAMPAIGN = "q1-clear-deepsets-hilar-10seed-20260903"
RELEASED_SHA256 = "15e456964c5f819aa882522a203946b515cc1034e217dfcbc474e5e50bf1719a"

BASE_ASSIGNMENTS = {
    "10110": (
        "subdiagnostic:52", "cpsc2018:52", "csn:53", "csn:56", "csn:59",
        "subdiagnostic:55", "subdiagnostic:58", "subdiagnostic:61",
        "rhythm:54", "rhythm:57", "rhythm:60",
        "superdiagnostic:53", "superdiagnostic:56", "superdiagnostic:59",
        "form:54", "form:57", "form:60", "cpsc2018:55", "cpsc2018:58", "cpsc2018:61",
    ),
    "10092": (
        "csn:54", "csn:52", "csn:57", "csn:60", "csn:61",
        "subdiagnostic:53", "subdiagnostic:56", "subdiagnostic:59",
        "rhythm:52", "rhythm:55", "rhythm:58",
        "superdiagnostic:54", "superdiagnostic:57", "superdiagnostic:60",
        "form:55", "form:58", "form:61", "cpsc2018:53", "cpsc2018:56", "cpsc2018:59",
    ),
    "10103": (
        "rhythm:53", "superdiagnostic:52", "form:52", "csn:55", "csn:58",
        "subdiagnostic:54", "subdiagnostic:57", "subdiagnostic:60",
        "rhythm:56", "rhythm:59", "rhythm:61",
        "superdiagnostic:55", "superdiagnostic:58", "superdiagnostic:61",
        "form:53", "form:56", "form:59", "cpsc2018:54", "cpsc2018:57", "cpsc2018:60",
    ),
}

ASSIGNMENTS = {
    node: tuple(f"{fraction}:{spec}" for fraction in FRACTIONS for spec in specs)
    for node, specs in BASE_ASSIGNMENTS.items()
}


def validate():
    expected = {f"{fraction}:{task}:{seed}" for fraction in FRACTIONS for task in TASKS for seed in SEEDS}
    actual = [spec for specs in ASSIGNMENTS.values() for spec in specs]
    assert len(actual) == 180 and len(set(actual)) == 180 and set(actual) == expected
    assert all(len(specs) == 60 for specs in ASSIGNMENTS.values())
    for seed in SEEDS:
        counts = {node: sum(spec.endswith(f":{seed}") for spec in specs) for node, specs in ASSIGNMENTS.items()}
        assert counts == {"10110": 6, "10092": 6, "10103": 6}


validate()
