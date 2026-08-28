"""Audit and analyze the strict paired DeepSets -> no-anchor HiLAR campaign.

The script consumes the 18 downloaded paired runs and uses the previously
frozen HUG prediction archives only as a source of audited patient/ECG row
identities.  Archived model scores are never used in the comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


TASKS = (
    "cpsc2018",
    "csn",
    "form",
    "rhythm",
    "subdiagnostic",
    "superdiagnostic",
)
SEEDS = (42, 43, 44)
METRICS = ("macro_auroc", "macro_auprc")
DATABASES = {
    "PTB-XL": ("form", "rhythm", "subdiagnostic", "superdiagnostic"),
    "CPSC2018": ("cpsc2018",),
    "CSN": ("csn",),
}
IDENTITY_NAME = re.compile(
    r"^(?P<task>cpsc2018|csn|form|rhythm|subdiagnostic|superdiagnostic)-"
    r"hug-seed(?P<seed>42|43|44)-(?:eval-best|full)$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def macro_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "macro_auroc": float(roc_auc_score(y_true, score, average="macro")),
        "macro_auprc": float(
            average_precision_score(y_true, score, average="macro")
        ),
    }


def all_classes_present(y_true: np.ndarray) -> bool:
    positives = y_true.sum(axis=0)
    return bool(np.all(positives > 0) and np.all(positives < len(y_true)))


def raw_two_sided_p(values: np.ndarray) -> float:
    lower = int(np.count_nonzero(values <= 0.0))
    upper = int(np.count_nonzero(values >= 0.0))
    return min(1.0, 2.0 * (min(lower, upper) + 1.0) / (len(values) + 1.0))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * p_values[key]))
        adjusted[key] = running
    return adjusted


def bootstrap_task(
    task: str,
    y_true: np.ndarray,
    patient_id: np.ndarray,
    baseline_score: np.ndarray,
    direct_score: np.ndarray,
    repetitions: int,
    seed: int,
    max_attempts: int,
) -> tuple[str, dict[str, Any]]:
    patients, inverse = np.unique(patient_id, return_inverse=True)
    rows_by_patient = [
        np.flatnonzero(inverse == index) for index in range(len(patients))
    ]
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in METRICS}
    attempts = 0
    missing_class = 0
    while len(values[METRICS[0]]) < repetitions:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"{task}: only {len(values[METRICS[0]])}/{repetitions} valid "
                f"resamples after {max_attempts} attempts"
            )
        chosen = rng.integers(0, len(patients), size=len(patients))
        selected = np.concatenate([rows_by_patient[index] for index in chosen])
        truth = y_true[selected]
        if not all_classes_present(truth):
            missing_class += 1
            continue
        baseline = macro_metrics(truth, baseline_score[selected])
        direct = macro_metrics(truth, direct_score[selected])
        for metric in METRICS:
            values[metric].append(direct[metric] - baseline[metric])
    arrays = {metric: np.asarray(items) for metric, items in values.items()}
    return task, {
        "unique_patients": int(len(patients)),
        "valid_repetitions": repetitions,
        "attempts": attempts,
        "rejected_missing_class": missing_class,
        "metrics": {
            metric: {
                "bootstrap_mean": float(array.mean()),
                "ci95": [float(value) for value in np.percentile(array, [2.5, 97.5])],
                "raw_two_sided_p": raw_two_sided_p(array),
            }
            for metric, array in arrays.items()
        },
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def discover_identities(root: Path) -> dict[tuple[str, int], Path]:
    found: dict[tuple[str, int], Path] = {}
    for path in root.rglob("test_predictions.npz"):
        match = IDENTITY_NAME.fullmatch(path.parent.name)
        if match is None:
            continue
        key = (match.group("task"), int(match.group("seed")))
        if key in found:
            raise ValueError(f"duplicate identity archive for {key}")
        found[key] = path
    expected = {(task, seed) for task in TASKS for seed in SEEDS}
    if set(found) != expected:
        raise ValueError(
            f"identity inventory mismatch: missing={sorted(expected-set(found))}, "
            f"unexpected={sorted(set(found)-expected)}"
        )
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--identity-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260828)
    parser.add_argument("--max-attempts", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.bootstrap != 10_000:
        raise ValueError("protocol requires exactly 10,000 bootstrap repetitions")

    identities = discover_identities(args.identity_root)
    audit: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    ensembles: dict[str, dict[str, np.ndarray]] = {}
    ensemble_rows: dict[str, dict[str, Any]] = {}

    baseline_hashes: set[str] = set()
    residual_hashes: set[str] = set()
    for task in TASKS:
        truths: list[np.ndarray] = []
        baseline_scores: list[np.ndarray] = []
        direct_scores: list[np.ndarray] = []
        identity_truths: list[np.ndarray] = []
        identity_ecg: list[np.ndarray] = []
        identity_patients: list[np.ndarray] = []
        for seed in SEEDS:
            run = args.campaign / f"{task}-seed{seed}"
            manifest_path = run / "paired-manifest.json"
            result_path = run / "formal-test/formal-test-result.json"
            prediction_path = run / "formal-test/test_predictions.npz"
            for path in (manifest_path, result_path, prediction_path):
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"missing artifact: {path}")
            manifest = load_json(manifest_path)
            result = load_json(result_path)
            manifest_checks = {
                "task": manifest.get("task") == task,
                "seed": manifest.get("seed") == seed,
                "pairing": manifest.get("pairing")
                == "DeepSets_s -> frozen logits_s -> zero-init no-anchor HiLAR_s",
                "feature_checkpoint_exact": manifest.get("feature_checkpoint_exact") is True,
                "anchor_lambda_zero": float(manifest.get("anchor_lambda", -1)) == 0.0,
                "test_not_used_for_selection": manifest.get("test_used_for_selection") is False,
                "baseline_path_seed": f"{task}-seed{seed}" in manifest.get("baseline_checkpoint", ""),
                "residual_path_seed": f"{task}-seed{seed}" in manifest.get("residual_checkpoint", ""),
            }
            result_checks = {
                "task": result.get("task") == task,
                "seed": result.get("seed") == seed,
                "access_count": result.get("access_count") == 1,
                "candidate": result.get("candidate") == "parameter-matched-direct",
                "test_not_used_for_selection": result.get("checkpoint_selection_used_test") is False,
            }
            if not all(manifest_checks.values()) or not all(result_checks.values()):
                raise ValueError(
                    f"{task} seed {seed}: provenance gate failed: "
                    f"{manifest_checks}, {result_checks}"
                )
            baseline_hash = str(manifest["baseline_checkpoint_sha256"])
            residual_hash = str(manifest["residual_checkpoint_sha256"])
            if len(baseline_hash) != 64 or len(residual_hash) != 64:
                raise ValueError(f"{task} seed {seed}: invalid checkpoint hash")
            baseline_hashes.add(baseline_hash)
            residual_hashes.add(residual_hash)

            with np.load(prediction_path, allow_pickle=False) as payload:
                required = {"y_true", "baseline_score", "direct_score", "delta_logits"}
                if not required.issubset(payload.files):
                    raise ValueError(f"{prediction_path}: missing {required-set(payload.files)}")
                truth = np.asarray(payload["y_true"])
                baseline_score = np.asarray(payload["baseline_score"], dtype=np.float64)
                direct_score = np.asarray(payload["direct_score"], dtype=np.float64)
                delta_logits = np.asarray(payload["delta_logits"], dtype=np.float64)
            if truth.ndim != 2 or baseline_score.shape != truth.shape or direct_score.shape != truth.shape:
                raise ValueError(f"{prediction_path}: shape mismatch")
            if delta_logits.shape != truth.shape:
                raise ValueError(f"{prediction_path}: delta shape mismatch")
            if not np.isin(truth, (0, 1)).all():
                raise ValueError(f"{prediction_path}: non-binary labels")

            identity_path = identities[(task, seed)]
            with np.load(identity_path, allow_pickle=False) as identity:
                identity_truth = np.asarray(identity["y_true"])
                ecg_id = np.asarray(identity["ecg_id"])
                patient_id = np.asarray(identity["patient_id"])
            if not np.array_equal(truth, identity_truth):
                raise ValueError(f"{task} seed {seed}: labels/order differ from identity archive")
            if len(ecg_id) != len(truth) or len(patient_id) != len(truth):
                raise ValueError(f"{task} seed {seed}: identity length mismatch")

            computed_baseline = macro_metrics(truth, baseline_score)
            computed_direct = macro_metrics(truth, direct_score)
            for model_name, computed, recorded in (
                ("baseline", computed_baseline, result["baseline"]),
                ("direct", computed_direct, result["direct"]),
            ):
                for short, full in (("auroc", "macro_auroc"), ("auprc", "macro_auprc")):
                    if not np.isclose(computed[full], float(recorded[short]), atol=1e-12):
                        raise ValueError(f"{task} seed {seed}: {model_name} {short} mismatch")
            seed_rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "baseline": computed_baseline,
                    "hilar": computed_direct,
                    "difference": {
                        metric: computed_direct[metric] - computed_baseline[metric]
                        for metric in METRICS
                    },
                }
            )
            audit.append(
                {
                    "task": task,
                    "seed": seed,
                    "manifest": str(manifest_path),
                    "manifest_sha256": sha256(manifest_path),
                    "formal_result_sha256": sha256(result_path),
                    "predictions_sha256": sha256(prediction_path),
                    "identity_source": str(identity_path),
                    "identity_sha256": sha256(identity_path),
                    "checks": {**manifest_checks, **{f"formal_{k}": v for k, v in result_checks.items()}},
                }
            )
            truths.append(truth)
            baseline_scores.append(baseline_score)
            direct_scores.append(direct_score)
            identity_truths.append(identity_truth)
            identity_ecg.append(ecg_id)
            identity_patients.append(patient_id)

        for name, arrays in (
            ("y_true", truths),
            ("identity_y_true", identity_truths),
            ("ecg_id", identity_ecg),
            ("patient_id", identity_patients),
        ):
            if any(not np.array_equal(arrays[0], item) for item in arrays[1:]):
                raise ValueError(f"{task}: {name} differs between seeds")
        if np.array_equal(baseline_scores[0], baseline_scores[1]) or np.array_equal(
            baseline_scores[0], baseline_scores[2]
        ):
            raise ValueError(f"{task}: independent DeepSets seed predictions are duplicated")
        ensemble_baseline = np.mean(np.stack(baseline_scores), axis=0)
        ensemble_direct = np.mean(np.stack(direct_scores), axis=0)
        baseline_metrics = macro_metrics(truths[0], ensemble_baseline)
        direct_metrics = macro_metrics(truths[0], ensemble_direct)
        ensemble_rows[task] = {
            "baseline": baseline_metrics,
            "hilar": direct_metrics,
            "difference": {
                metric: direct_metrics[metric] - baseline_metrics[metric]
                for metric in METRICS
            },
        }
        ensembles[task] = {
            "y_true": truths[0],
            "patient_id": identity_patients[0],
            "baseline_score": ensemble_baseline,
            "direct_score": ensemble_direct,
        }

    if len(baseline_hashes) != 18 or len(residual_hashes) != 18:
        raise ValueError("checkpoint hashes are not unique across all 18 paired runs")

    task_seed_means: dict[str, Any] = {}
    for task in TASKS:
        rows = [row for row in seed_rows if row["task"] == task]
        task_seed_means[task] = {
            model: mean_metrics([row[model] for row in rows])
            for model in ("baseline", "hilar", "difference")
        }
    six_task_seed_mean = {
        model: mean_metrics([task_seed_means[task][model] for task in TASKS])
        for model in ("baseline", "hilar", "difference")
    }
    database_seed_means: dict[str, Any] = {}
    for database, tasks in DATABASES.items():
        database_seed_means[database] = {
            model: mean_metrics([task_seed_means[task][model] for task in tasks])
            for model in ("baseline", "hilar", "difference")
        }
    database_equal_seed_mean = {
        model: mean_metrics([database_seed_means[name][model] for name in DATABASES])
        for model in ("baseline", "hilar", "difference")
    }
    six_task_ensemble_mean = {
        model: mean_metrics([ensemble_rows[task][model] for task in TASKS])
        for model in ("baseline", "hilar", "difference")
    }
    database_ensemble_means: dict[str, Any] = {}
    for database, tasks in DATABASES.items():
        database_ensemble_means[database] = {
            model: mean_metrics([ensemble_rows[task][model] for task in tasks])
            for model in ("baseline", "hilar", "difference")
        }
    database_equal_ensemble_mean = {
        model: mean_metrics([database_ensemble_means[name][model] for name in DATABASES])
        for model in ("baseline", "hilar", "difference")
    }

    child_seeds = np.random.SeedSequence(args.bootstrap_seed).spawn(len(TASKS))
    bootstrap: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                bootstrap_task,
                task,
                ensembles[task]["y_true"],
                ensembles[task]["patient_id"],
                ensembles[task]["baseline_score"],
                ensembles[task]["direct_score"],
                args.bootstrap,
                int(child_seeds[index].generate_state(1)[0]),
                args.max_attempts,
            ): task
            for index, task in enumerate(TASKS)
        }
        for future in as_completed(futures):
            task, row = future.result()
            bootstrap[task] = row
            print(f"bootstrap complete: {task}", flush=True)

    raw_p = {
        f"{task}:{metric}": bootstrap[task]["metrics"][metric]["raw_two_sided_p"]
        for task in TASKS
        for metric in METRICS
    }
    adjusted = holm_adjust(raw_p)
    for key, value in adjusted.items():
        task, metric = key.split(":", 1)
        bootstrap[task]["metrics"][metric]["holm_adjusted_p"] = value

    positive = {
        "auroc": sum(row["difference"]["macro_auroc"] > 0 for row in seed_rows),
        "auprc": sum(row["difference"]["macro_auprc"] > 0 for row in seed_rows),
        "both": sum(
            row["difference"]["macro_auroc"] > 0
            and row["difference"]["macro_auprc"] > 0
            for row in seed_rows
        ),
        "total": len(seed_rows),
    }
    output = {
        "schema_version": 1,
        "status": "complete",
        "scope": "strict paired DeepSets seed -> no-anchor HiLAR seed formal test",
        "protocol": {
            "tasks": list(TASKS),
            "seeds": list(SEEDS),
            "ensemble": "arithmetic mean of three seed probability arrays",
            "bootstrap": "10,000 valid patient-cluster resamples per task",
            "bootstrap_seed": args.bootstrap_seed,
            "ci": "two-sided percentile 95%",
            "holm": "single family of 12 task x metric hypotheses",
            "identity_only_archive_use": True,
            "archived_model_scores_used": False,
        },
        "provenance_audit": {
            "passed": True,
            "unique_baseline_checkpoint_hashes": len(baseline_hashes),
            "unique_residual_checkpoint_hashes": len(residual_hashes),
            "artifacts": audit,
        },
        "seed_results": seed_rows,
        "positive_counts": positive,
        "task_three_seed_means": task_seed_means,
        "six_task_three_seed_mean": six_task_seed_mean,
        "database_three_seed_means": database_seed_means,
        "three_database_equal_seed_mean": database_equal_seed_mean,
        "three_seed_probability_ensemble": ensemble_rows,
        "six_task_ensemble_mean": six_task_ensemble_mean,
        "database_ensemble_means": database_ensemble_means,
        "three_database_equal_ensemble_mean": database_equal_ensemble_mean,
        "patient_level_bootstrap": {task: bootstrap[task] for task in TASKS},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    incoming = args.output.with_suffix(args.output.suffix + ".incoming")
    incoming.write_text(json.dumps(output, indent=2), encoding="utf-8")
    os.replace(incoming, args.output)

    lines = [
        "# Strict paired DeepSets → no-anchor HiLAR results",
        "",
        "All 18 task-seed runs passed provenance, label, row-order, and metric audits.",
        "Archived prediction files were used only for patient/ECG identities; archived scores were not used.",
        "",
        "## Three-seed task means",
        "",
        "| Task | DeepSets AUROC/AUPRC | HiLAR AUROC/AUPRC | Delta pp |",
        "|---|---:|---:|---:|",
    ]
    for task in TASKS:
        row = task_seed_means[task]
        lines.append(
            f"| {task} | {row['baseline']['macro_auroc']:.6f} / {row['baseline']['macro_auprc']:.6f} "
            f"| {row['hilar']['macro_auroc']:.6f} / {row['hilar']['macro_auprc']:.6f} "
            f"| {100*row['difference']['macro_auroc']:+.3f} / {100*row['difference']['macro_auprc']:+.3f} |"
        )
    lines += [
        "",
        "## Patient bootstrap on three-seed probability ensembles",
        "",
        "| Task | Metric | Point delta pp | 95% CI pp | Holm p |",
        "|---|---|---:|---:|---:|",
    ]
    for task in TASKS:
        for metric in METRICS:
            row = bootstrap[task]["metrics"][metric]
            point = ensemble_rows[task]["difference"][metric]
            lines.append(
                f"| {task} | {metric} | {100*point:+.3f} | "
                f"[{100*row['ci95'][0]:+.3f}, {100*row['ci95'][1]:+.3f}] | "
                f"{row['holm_adjusted_p']:.6g} |"
            )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
