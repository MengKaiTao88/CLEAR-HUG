"""Audit and analyze the frozen ten-seed three-model formal campaign."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from protocol import SEEDS, TASKS

MODELS = ("hug", "deepsets", "hilar")
METRICS = ("macro_auroc", "macro_auprc")
COMPARISONS = {"deepsets_minus_hug": ("deepsets", "hug"), "hilar_minus_deepsets": ("hilar", "deepsets"), "hilar_minus_hug": ("hilar", "hug")}
DATABASES = {"PTB-XL": ("superdiagnostic", "subdiagnostic", "form", "rhythm"), "CPSC2018": ("cpsc2018",), "CSN": ("csn",)}


def metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    valid = [i for i in range(y.shape[1]) if np.unique(y[:, i]).size == 2]
    return {"macro_auroc": float(roc_auc_score(y[:, valid], score[:, valid], average="macro")), "macro_auprc": float(average_precision_score(y[:, valid], score[:, valid], average="macro"))}


def holm(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=values.get); result = {}; running = 0.0
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * values[key])); result[key] = running
    return result


def bootstrap(task, y, patient_id, high, low, repetitions, seed):
    patients, inverse = np.unique(patient_id, return_inverse=True)
    rows = [np.flatnonzero(inverse == i) for i in range(len(patients))]
    rng = np.random.default_rng(seed); values = {metric: [] for metric in METRICS}; attempts = 0
    while len(values[METRICS[0]]) < repetitions:
        attempts += 1
        if attempts > 1_000_000: raise RuntimeError(f"{task}: bootstrap exhausted")
        selected = np.concatenate([rows[i] for i in rng.integers(0, len(patients), len(patients))])
        truth = y[selected]; positives = truth.sum(0)
        if np.any(positives == 0) or np.any(positives == len(truth)): continue
        a, b = metrics(truth, high[selected]), metrics(truth, low[selected])
        for metric in METRICS: values[metric].append(a[metric] - b[metric])
    output = {}
    for metric, raw in values.items():
        array = np.asarray(raw); lower = np.count_nonzero(array <= 0); upper = np.count_nonzero(array >= 0)
        output[metric] = {"mean": float(array.mean()), "ci95": np.percentile(array, [2.5, 97.5]).tolist(), "raw_p": min(1.0, 2.0 * (min(lower, upper) + 1) / (len(array) + 1))}
    return task, output


def identity_for(task: str, truth: np.ndarray, root: Path) -> np.ndarray:
    for path in root.rglob("test_predictions.npz"):
        if task not in path.as_posix().lower(): continue
        with np.load(path, allow_pickle=False) as payload:
            if "y_true" not in payload or not np.array_equal(payload["y_true"], truth): continue
            for key in ("patient_id", "patient_ids", "ecg_id", "ecg_ids"):
                if key in payload and len(payload[key]) == len(truth): return np.asarray(payload[key])
    raise RuntimeError(f"no audited identity array for {task}")


def summarize(rows):
    return {model: {metric: {"mean": float(np.mean([r[model][metric] for r in rows])), "sample_sd": float(np.std([r[model][metric] for r in rows], ddof=1))} for metric in METRICS} for model in MODELS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True); parser.add_argument("--identity-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--bootstrap", type=int, default=10_000); parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.bootstrap != 10_000 or args.output.exists(): raise RuntimeError("frozen bootstrap count or output violation")
    seed_rows = []; ensembles = {}; identities = {}
    expected_hashes = {model: set() for model in MODELS}
    for task, config in TASKS.items():
        scores = {model: [] for model in MODELS}; truths = []
        for seed in SEEDS:
            run = args.campaign / f"{task}-seed{seed}"
            manifest = json.loads((run / "train-val-complete.json").read_text())
            complete = json.loads((run / "formal-test/complete.json").read_text())
            if manifest["selection_metric"] != "macro_auroc" or manifest["test_used_for_selection"] is not False or complete["test_used_for_selection"] is not False: raise RuntimeError(f"protocol violation {task}:{seed}")
            with np.load(run / "formal-test/paired/test_predictions.npz") as p, np.load(run / "formal-test/hug/test_predictions.npz") as h:
                y = np.asarray(p["y_true"]); hy = np.asarray(h["y_true"])
                if y.shape != tuple(config["test_shape"]) or not np.array_equal(y, hy): raise RuntimeError(f"shape/order mismatch {task}:{seed}")
                current = {"hug": np.asarray(h["hug_score"]), "deepsets": np.asarray(p["baseline_score"]), "hilar": np.asarray(p["direct_score"])}
            computed = {model: metrics(y, score) for model, score in current.items()}
            for model in MODELS:
                recorded = complete[model]
                for metric in METRICS:
                    key = metric if metric in recorded else metric.replace("macro_", "")
                    if abs(computed[model][metric] - float(recorded[key])) > 1e-10: raise RuntimeError(f"metric mismatch {task}:{seed}:{model}")
                scores[model].append(current[model]); expected_hashes[model].add(manifest[f"{model}_checkpoint_sha256"])
            truths.append(y); seed_rows.append({"task": task, "seed": seed, **computed, "differences": {name: {metric: computed[hi][metric] - computed[lo][metric] for metric in METRICS} for name, (hi, lo) in COMPARISONS.items()}})
        if any(not np.array_equal(truths[0], item) for item in truths[1:]): raise RuntimeError(f"truth differs across seeds: {task}")
        ensemble_scores = {model: np.mean(np.stack(values), axis=0) for model, values in scores.items()}
        ensembles[task] = {"y_true": truths[0], **ensemble_scores, "metrics": {model: metrics(truths[0], score) for model, score in ensemble_scores.items()}}
        identities[task] = identity_for(task, truths[0], args.identity_root)
    if any(len(values) != 60 for values in expected_hashes.values()): raise RuntimeError({k: len(v) for k, v in expected_hashes.items()})
    task_summary = {task: summarize([row for row in seed_rows if row["task"] == task]) for task in TASKS}
    task_equal = {model: {metric: float(np.mean([task_summary[t][model][metric]["mean"] for t in TASKS])) for metric in METRICS} for model in MODELS}
    database = {db: {model: {metric: float(np.mean([task_summary[t][model][metric]["mean"] for t in tasks])) for metric in METRICS} for model in MODELS} for db, tasks in DATABASES.items()}
    database_equal = {model: {metric: float(np.mean([database[db][model][metric] for db in DATABASES])) for metric in METRICS} for model in MODELS}
    boot = {}
    for comparison_index, (name, (high, low)) in enumerate(COMPARISONS.items()):
        futures = {}; rows = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for task_index, task in enumerate(TASKS):
                e = ensembles[task]; futures[pool.submit(bootstrap, task, e["y_true"], identities[task], e[high], e[low], args.bootstrap, 20260903 + comparison_index * 100 + task_index)] = task
            for future in as_completed(futures): task, result = future.result(); rows[task] = result
        adjusted = holm({f"{task}:{metric}": rows[task][metric]["raw_p"] for task in TASKS for metric in METRICS})
        for key, value in adjusted.items(): task, metric = key.split(":"); rows[task][metric]["holm_p"] = value
        boot[name] = rows
    positive = {name: {metric: sum(row["differences"][name][metric] > 0 for row in seed_rows) for metric in METRICS} for name in COMPARISONS}
    args.output.mkdir(parents=True)
    payload = {"protocol": {"seeds": list(SEEDS), "bootstrap": 10_000, "archived_scores_used": False}, "seed_results": seed_rows, "task_10seed_mean_sd": task_summary, "six_task_equal_mean": task_equal, "database_means": database, "three_database_equal_mean": database_equal, "positive_counts_out_of_60": positive, "ensemble_metrics": {task: e["metrics"] for task, e in ensembles.items()}, "patient_bootstrap": boot}
    (args.output / "final-analysis.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (args.output / "seed-results.csv").open("w", newline="") as handle:
        fields = ["task", "seed"] + [f"{m}_{x}" for m in MODELS for x in METRICS]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in seed_rows: writer.writerow({"task": row["task"], "seed": row["seed"], **{f"{m}_{x}": row[m][x] for m in MODELS for x in METRICS}})
    print(json.dumps({"task_10seed_mean_sd": task_summary, "positive_counts_out_of_60": positive}, indent=2))


if __name__ == "__main__":
    main()
