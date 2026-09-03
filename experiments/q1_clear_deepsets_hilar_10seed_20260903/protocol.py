"""Frozen protocol constants for the 2026-09-03 ten-seed campaign."""

from __future__ import annotations

TASKS = {
    "superdiagnostic": {"classes": 5, "dataset": "PTBXL_QRS/superdiagnostic", "test_shape": (2158, 5)},
    "subdiagnostic": {"classes": 23, "dataset": "PTBXL_QRS/subdiagnostic", "test_shape": (2158, 23)},
    "form": {"classes": 19, "dataset": "PTBXL_QRS/form", "test_shape": (880, 19)},
    "rhythm": {"classes": 12, "dataset": "PTBXL_QRS/rhythm", "test_shape": (2098, 12)},
    "cpsc2018": {"classes": 9, "dataset": "CPSC2018_QRS/data", "test_shape": (1376, 9)},
    "csn": {"classes": 38, "dataset": "CSN_QRS/data", "test_shape": (4620, 38)},
}

SEEDS = tuple(range(42, 52))
CAMPAIGN = "q1-clear-deepsets-hilar-10seed-20260903"
RELEASED_SHA256 = "15e456964c5f819aa882522a203946b515cc1034e217dfcbc474e5e50bf1719a"

ASSIGNMENTS = {
    "10110": {
        "root": "/root/107552503710",
        "specs": (
            "subdiagnostic:42", "cpsc2018:42",
            "csn:43", "csn:46", "csn:49",
            "subdiagnostic:45", "subdiagnostic:48", "subdiagnostic:51",
            "rhythm:44", "rhythm:47", "rhythm:50",
            "superdiagnostic:43", "superdiagnostic:46", "superdiagnostic:49",
            "form:44", "form:47", "form:50",
            "cpsc2018:45", "cpsc2018:48", "cpsc2018:51",
        ),
        "canaries": ("subdiagnostic:42", "cpsc2018:42"),
    },
    "10092": {
        "root": "/root/107552503710-1",
        "specs": (
            "csn:44", "csn:42",
            "csn:47", "csn:50", "csn:51",
            "subdiagnostic:43", "subdiagnostic:46", "subdiagnostic:49",
            "rhythm:42", "rhythm:45", "rhythm:48",
            "superdiagnostic:44", "superdiagnostic:47", "superdiagnostic:50",
            "form:45", "form:48", "form:51",
            "cpsc2018:43", "cpsc2018:46", "cpsc2018:49",
        ),
        "canaries": ("rhythm:42", "cpsc2018:43"),
    },
    "10103": {
        "root": "/root/107552503710-2",
        "specs": (
            "rhythm:43", "superdiagnostic:42",
            "form:42", "csn:45", "csn:48",
            "subdiagnostic:44", "subdiagnostic:47", "subdiagnostic:50",
            "rhythm:46", "rhythm:49", "rhythm:51",
            "superdiagnostic:45", "superdiagnostic:48", "superdiagnostic:51",
            "form:43", "form:46", "form:49",
            "cpsc2018:44", "cpsc2018:47", "cpsc2018:50",
        ),
        "canaries": ("rhythm:43", "superdiagnostic:42"),
    },
}


def validate_protocol() -> None:
    expected = {f"{task}:{seed}" for task in TASKS for seed in SEEDS}
    assigned = [spec for node in ASSIGNMENTS.values() for spec in node["specs"]]
    if len(assigned) != 60 or set(assigned) != expected or len(set(assigned)) != 60:
        raise RuntimeError("assignment must cover every task-seed exactly once")
    for node, payload in ASSIGNMENTS.items():
        if len(payload["specs"]) != 20:
            raise RuntimeError(f"{node} does not have 20 specs")
        if not set(payload["canaries"]).issubset(payload["specs"]):
            raise RuntimeError(f"{node} canaries are not assigned to that node")
    for seed in SEEDS:
        counts = {node: 0 for node in ASSIGNMENTS}
        for node, payload in ASSIGNMENTS.items():
            counts[node] = sum(spec.endswith(f":{seed}") for spec in payload["specs"])
        if set(counts.values()) != {2}:
            raise RuntimeError(f"seed {seed} is hardware-confounded: {counts}")


validate_protocol()
