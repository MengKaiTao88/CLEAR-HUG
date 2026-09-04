"""Plot a strictly aligned HUG -> DeepSets -> HiLAR decision-space comparison.

All panels use the same single-label PTB-XL records in the same order.  HiLAR
does not expose a natural fused latent with the same semantics/dimensionality
as the other models, so the honest common representation is the class-logit
(decision) space.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


CLASS_NAMES = ("CD", "HYP", "MI", "NORM", "STTC")
DISPLAY_ORDER = ("NORM", "MI", "STTC", "CD", "HYP")
COLORS = {
    "NORM": "#E8A6B5",
    "MI": "#E67E22",
    "STTC": "#2878B5",
    "CD": "#3B9348",
    "HYP": "#76569A",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_logit(score: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(score, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(clipped) - np.log1p(-clipped)


def load_inputs(hug_path: Path, paired_path: Path):
    with np.load(hug_path, allow_pickle=False) as hug, np.load(
        paired_path, allow_pickle=False
    ) as paired:
        hug_truth = np.asarray(hug["y_true"])
        paired_truth = np.asarray(paired["y_true"])
        if not np.array_equal(hug_truth, paired_truth):
            raise ValueError("HUG and paired labels/order are not identical")
        if "hug_logits" in hug.files:
            hug_logits = np.asarray(hug["hug_logits"], dtype=np.float64)
            hug_source = "hug_logits"
        else:
            score_key = "hug_score" if "hug_score" in hug.files else "y_score"
            hug_logits = stable_logit(hug[score_key])
            hug_source = f"inverse_sigmoid({score_key})"
        if "baseline_logits" in paired.files:
            deepsets_logits = np.asarray(paired["baseline_logits"], dtype=np.float64)
        else:
            deepsets_logits = stable_logit(paired["baseline_score"])
        if "direct_logits" in paired.files:
            hilar_logits = np.asarray(paired["direct_logits"], dtype=np.float64)
        elif "direct_score" in paired.files:
            hilar_logits = stable_logit(paired["direct_score"])
        else:
            hilar_logits = deepsets_logits + np.asarray(
                paired["delta_logits"], dtype=np.float64
            )
    expected = hug_truth.shape
    for name, values in (
        ("HUG", hug_logits),
        ("DeepSets", deepsets_logits),
        ("HiLAR", hilar_logits),
    ):
        if values.shape != expected:
            raise ValueError(f"{name} shape {values.shape} != labels {expected}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite logits")
    return hug_truth, (hug_logits, deepsets_logits, hilar_logits), hug_source


def embed_and_score(values: np.ndarray, labels: np.ndarray, seed: int):
    standardized = StandardScaler().fit_transform(values)
    embedded = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        max_iter=2000,
        random_state=seed,
    ).fit_transform(standardized)
    return embedded, {
        "silhouette": float(silhouette_score(standardized, labels)),
        "davies_bouldin": float(davies_bouldin_score(standardized, labels)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hug", type=Path, required=True)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    truth, all_logits, hug_source = load_inputs(args.hug, args.paired)
    single_mask = np.isclose(truth.sum(axis=1), 1.0)
    selected_indices = np.flatnonzero(single_mask)
    single_truth = truth[single_mask]
    label_indices = single_truth.argmax(axis=1)
    label_names = np.asarray(CLASS_NAMES)[label_indices]

    titles = ("HUG", "Identity-aware DeepSets", "HiLAR")
    embeddings, metrics = [], []
    for values in all_logits:
        embedding, result = embed_and_score(values[single_mask], label_indices, args.seed)
        embeddings.append(embedding)
        metrics.append(result)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
    handles = []
    for panel, (axis, title, embedding, result) in enumerate(
        zip(axes, titles, embeddings, metrics)
    ):
        for class_name in DISPLAY_ORDER:
            mask = label_names == class_name
            artist = axis.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=13,
                c=COLORS[class_name],
                alpha=0.72,
                linewidths=0,
                rasterized=True,
                label=class_name,
            )
            if panel == 0:
                handles.append(artist)
        axis.set_title(
            f"({chr(97 + panel)}) {title}\n"
            f"Silhouette = {result['silhouette']:.3f}   DBI = {result['davies_bouldin']:.3f}",
            pad=9,
        )
        axis.set_xlabel("t-SNE 1")
        if panel == 0:
            axis.set_ylabel("t-SNE 2")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_color("#555555")
            spine.set_linewidth(0.8)
    fig.legend(
        handles,
        DISPLAY_ORDER,
        title="PTB-XL superclass",
        loc="center left",
        bbox_to_anchor=(0.865, 0.5),
        frameon=True,
        edgecolor="#BBBBBB",
    )
    fig.suptitle(
        "Representation evolution under different cross-lead modeling strategies",
        fontsize=14,
        y=0.975,
    )
    fig.subplots_adjust(left=0.055, right=0.855, bottom=0.12, top=0.80, wspace=0.025)
    base = args.output_dir / "hug_deepsets_hilar_decision_space"
    fig.savefig(base.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    report = {
        "schema_version": 1,
        "dataset": "PTB-XL",
        "task": "superdiagnostic",
        "split": "test",
        "model_seed": args.seed,
        "representation": "five-dimensional class-logit decision space",
        "class_order_in_files": list(CLASS_NAMES),
        "display_order": list(DISPLAY_ORDER),
        "records_total": int(len(truth)),
        "records_single_label": int(single_mask.sum()),
        "selected_indices_sha256": hashlib.sha256(
            selected_indices.astype("<i8").tobytes()
        ).hexdigest(),
        "class_counts": {
            name: int((label_names == name).sum()) for name in DISPLAY_ORDER
        },
        "alignment": {
            "labels_equal": True,
            "record_order_equal": True,
            "same_single_label_indices_all_panels": True,
        },
        "inputs": {
            "hug": str(args.hug.resolve()),
            "hug_sha256": sha256(args.hug),
            "hug_logit_source": hug_source,
            "paired": str(args.paired.resolve()),
            "paired_sha256": sha256(args.paired),
        },
        "tsne": {
            "dimensions": 2,
            "perplexity": 30,
            "learning_rate": "auto",
            "init": "pca",
            "max_iter": 2000,
            "random_state": args.seed,
            "preprocessing": "per-panel StandardScaler on 5D logits; no PCA because source dimension is 5",
        },
        "cluster_metrics_space": "per-panel standardized 5D logits (not t-SNE coordinates)",
        "metrics": {title: result for title, result in zip(titles, metrics)},
        "interpretation_guardrail": (
            "t-SNE is descriptive; visual separation is not evidence of monotonic model improvement."
        ),
    }
    (base.with_suffix(".json")).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with base.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("model", "silhouette", "davies_bouldin", "records"),
        )
        writer.writeheader()
        for title, result in zip(titles, metrics):
            writer.writerow({"model": title, **result, "records": int(single_mask.sum())})
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
