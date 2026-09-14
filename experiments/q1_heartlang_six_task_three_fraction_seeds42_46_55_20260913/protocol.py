"""Frozen protocol for the HeartLang/ST-MEM linear-probe benchmark."""

CAMPAIGN = "q1-heartlang-stmem-linear-probe-seeds42-46-55-20260914"
INPUT_CAMPAIGN = "q1-clear-deepsets-hilar-10seed-20260903"
SEEDS = (42, 46, 55)
FRACTIONS = {"100pct": 1.0, "10pct": 0.10, "1pct": 0.01}
MODELS = ("heartlang", "stmem")
TASKS = {
    "superdiagnostic": (5, "PTBXL_QRS/superdiagnostic", "PTBXL/superdiagnostic/data", 17084),
    "subdiagnostic": (23, "PTBXL_QRS/subdiagnostic", "PTBXL/subdiagnostic/data", 17084),
    "form": (19, "PTBXL_QRS/form", "PTBXL/form/data", 7197),
    "rhythm": (12, "PTBXL_QRS/rhythm", "PTBXL/rhythm/data", 16832),
    "cpsc2018": (9, "CPSC2018_QRS/data", "CPSC2018/data", 4950),
    "csn": (38, "CSN_QRS/data", "CSN/data", 16546),
}
# Each node receives two tasks. Every seed/fraction/model remains on the same
# node for a task so frozen feature caches are computed exactly once.
NODE_TASKS = {
    "10110": ("subdiagnostic", "rhythm"),
    "10092": ("csn", "cpsc2018"),
    "10103": ("superdiagnostic", "form"),
}
NODES = {
    "10110": (".env.remote", "10.109.118.172", 10110, "/root/107552503710"),
    "10092": (".env.remote.secondary", "10.109.118.204", 10092, "/root/107552503710-1"),
    "10103": (".env.remote.tertiary", "10.109.118.204", 10103, "/root/107552503710-2"),
}
CANARIES = {
    "10110": ("heartlang", "100pct", "subdiagnostic", 42),
    "10092": ("stmem", "10pct", "cpsc2018", 46),
    "10103": ("heartlang", "1pct", "form", 55),
}
HEARTLANG_COMMIT = "a08afd8117e813fee212a1ac283db4a7140a4669"
HEARTLANG_SHA256 = "9dc89d72a3f1cacd941892103eaa523d44e373164cd4fadfb975907cc65864f6"
HEARTLANG_VQHBR_SHA256 = "c510f170c34cc134dcea61c92d11cd9d117dbe6d80c6662e72d3276011397490"
STMEM_COMMIT = "311c6446894dee4126db8d2520024e9f62cf1616"
STMEM_SHA256 = "6f63d370fd1a2ccf2fbddd681571adf985eeae3f652973f0071342a39b7793d6"


def specs(node: str):
    return tuple(
        (model, fraction, task, seed)
        for task in NODE_TASKS[node]
        for fraction in FRACTIONS
        for seed in SEEDS
        for model in MODELS
    )


def validate() -> None:
    task_sets = [set(value) for value in NODE_TASKS.values()]
    assert set().union(*task_sets) == set(TASKS)
    assert sum(map(len, task_sets)) == len(set().union(*task_sets))
    actual = [item for node in NODES for item in specs(node)]
    expected = [
        (model, fraction, task, seed)
        for model in MODELS
        for fraction in FRACTIONS
        for task in TASKS
        for seed in SEEDS
    ]
    assert len(actual) == 108 and len(set(actual)) == 108
    assert set(actual) == set(expected)
    assert all(len(specs(node)) == 36 for node in NODES)


validate()
