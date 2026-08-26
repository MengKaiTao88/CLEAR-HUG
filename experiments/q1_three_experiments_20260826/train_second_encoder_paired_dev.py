"""Validation-only paired DeepSets versus HiLAR on frozen HeartLLM features.

The earlier ``train_second_encoder_dev.py`` was an exploratory HeartLLM plus
attention head.  This script implements the intended encoder-transfer test:
the HeartLLM encoder is held fixed and the same lead-wise features are used to
train a DeepSets baseline followed by a zero-initialized, anchored local
residual (HiLAR/Final Direct) head.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import atomic_json, macro_metrics, seed_all


class HeartLLMFeatureDataset(Dataset):
    """Memory-mapped HeartLLM per-lead features and PTB-XL labels."""

    def __init__(self, root: Path):
        self.features = np.load(root / "features.npy", mmap_mode="r")
        self.labels = np.load(root / "labels.npy", mmap_mode="r")
        if self.features.ndim != 3 or self.features.shape[1:] != (12, 32):
            raise RuntimeError(
                f"expected HeartLLM features with shape (N, 12, 32), got {self.features.shape}"
            )
        if len(self.features) != len(self.labels):
            raise RuntimeError("HeartLLM cache arrays have inconsistent lengths")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            np.asarray(self.features[index], dtype=np.float32),
            np.asarray(self.labels[index], dtype=np.float32),
        )


class HeartLLMDeepSets(nn.Module):
    """DeepSets over the twelve observed HeartLLM lead features."""

    def __init__(
        self,
        classes: int,
        input_dim: int = 32,
        identity_dim: int = 32,
        hidden: int = 192,
        dim: int = 128,
    ):
        super().__init__()
        self.lead_embedding = nn.Embedding(12, identity_dim)
        self.phi = nn.Sequential(
            nn.Linear(input_dim + identity_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.rho = nn.Sequential(nn.Linear(hidden, dim), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, classes))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, leads, _ = features.shape
        lead_ids = torch.arange(leads, device=features.device)
        identity = self.lead_embedding(lead_ids)[None].expand(batch, -1, -1)
        values = self.phi(torch.cat([features, identity], dim=-1))
        pooled = values.mean(dim=1)
        return self.head(self.rho(pooled))


class HiLARResidual(nn.Module):
    """Zero-initialized local residual correction anchored to DeepSets logits."""

    def __init__(
        self,
        classes: int,
        input_dim: int = 32,
        identity_dim: int = 32,
        hidden: int = 128,
        residual_dim: int = 32,
    ):
        super().__init__()
        self.lead_embedding = nn.Embedding(12, identity_dim)
        self.direct = nn.Sequential(
            nn.Linear(input_dim + identity_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        self.encoder = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 64, bias=False),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, residual_dim, bias=False),
        )
        self.delta_head = nn.Linear(residual_dim, classes)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(
        self, features: torch.Tensor, baseline_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, leads, _ = features.shape
        lead_ids = torch.arange(leads, device=features.device)
        identity = self.lead_embedding(lead_ids)[None].expand(batch, -1, -1)
        tokens = self.encoder(self.direct(torch.cat([features, identity], dim=-1)))
        summary = tokens.mean(dim=1)
        delta = self.delta_head(summary)
        return baseline_logits + delta, delta


def autocast_context(device: torch.device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


@torch.inference_mode()
def evaluate_deepsets(model, loader, device):
    model.eval()
    truths, scores = [], []
    for features, labels in loader:
        with autocast_context(device):
            logits = model(features.to(device, non_blocking=True))
        truths.append(labels.numpy())
        scores.append(torch.sigmoid(logits).float().cpu().numpy())
    truth, score = np.concatenate(truths), np.concatenate(scores)
    return macro_metrics(truth, score), truth, score


@torch.inference_mode()
def evaluate_hilar(model, baseline, loader, device):
    model.eval()
    baseline.eval()
    truths, scores, deltas = [], [], []
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        with autocast_context(device):
            base_logits = baseline(features)
            logits, delta = model(features, base_logits)
        truths.append(labels.numpy())
        scores.append(torch.sigmoid(logits).float().cpu().numpy())
        deltas.append(delta.float().cpu().numpy())
    truth, score, delta = np.concatenate(truths), np.concatenate(scores), np.concatenate(deltas)
    metrics = macro_metrics(truth, score)
    metrics["mean_absolute_delta_logit"] = float(np.abs(delta).mean())
    return metrics, truth, score


def train_deepsets(args, train_loader, val_loader, train_data, val_data, device):
    output = args.output / "deepsets"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "complete.json").is_file():
        return json.loads((output / "complete.json").read_text(encoding="utf-8"))

    model = HeartLLMDeepSets(args.classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best = -math.inf
    best_epoch = None
    best_metrics = None
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for features, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                logits = model(features.to(device, non_blocking=True))
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, labels.to(device, non_blocking=True)
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        validation, truth, score = evaluate_deepsets(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **validation}
        history.append(row)
        print(json.dumps({"model": "deepsets", **row}), flush=True)
        if validation["macro_auroc"] > best:
            best = validation["macro_auroc"]
            best_epoch = epoch
            best_metrics = dict(validation)
            torch.save({"model": model.state_dict(), "epoch": epoch}, output / "checkpoint-best.pth")
            np.savez_compressed(output / "val_predictions.npz", y_true=truth, y_score=score)

    payload = {
        "schema_version": 1,
        "status": "complete",
        "scope": "development train/validation only",
        "formal_test_used": False,
        "model": "DeepSets",
        "encoder": "HeartLLM ECGEncoder (frozen)",
        "task": args.task,
        "seed": args.seed,
        "features": str(args.features),
        "best_validation": {
            "macro_auroc": best,
            "macro_auprc": float(best_metrics["macro_auprc"]),
            "selection_metric": "Macro AUROC",
            "epoch": best_epoch,
        },
        "epochs": history,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output / "complete.json", payload)
    return payload


def train_hilar(args, train_loader, val_loader, train_data, val_data, device):
    output = args.output / "hilar"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "complete.json").is_file():
        return json.loads((output / "complete.json").read_text(encoding="utf-8"))

    baseline = HeartLLMDeepSets(args.classes).to(device)
    baseline_payload = torch.load(
        args.output / "deepsets" / "checkpoint-best.pth", map_location="cpu"
    )
    baseline.load_state_dict(baseline_payload.get("model", baseline_payload), strict=True)
    baseline.eval()
    for parameter in baseline.parameters():
        parameter.requires_grad = False

    model = HiLARResidual(args.classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best = -math.inf
    best_epoch = None
    best_metrics = None
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for features, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            features = features.to(device, non_blocking=True)
            with torch.no_grad():
                with autocast_context(device):
                    base_logits = baseline(features)
            with autocast_context(device):
                logits, delta = model(features, base_logits)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, labels.to(device, non_blocking=True)
                )
                loss = loss + float(args.anchor_lambda) * delta.square().mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        validation, truth, score = evaluate_hilar(model, baseline, val_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **validation}
        history.append(row)
        print(json.dumps({"model": "hilar", **row}), flush=True)
        if validation["macro_auroc"] > best:
            best = validation["macro_auroc"]
            best_epoch = epoch
            best_metrics = dict(validation)
            torch.save({"model": model.state_dict(), "epoch": epoch}, output / "checkpoint-best.pth")
            np.savez_compressed(output / "val_predictions.npz", y_true=truth, y_score=score)

    payload = {
        "schema_version": 1,
        "status": "complete",
        "scope": "development train/validation only",
        "formal_test_used": False,
        "model": "HiLAR / Final Direct residual",
        "encoder": "HeartLLM ECGEncoder (frozen)",
        "task": args.task,
        "seed": args.seed,
        "features": str(args.features),
        "anchor_lambda": float(args.anchor_lambda),
        "zero_initialized_delta_head": True,
        "best_validation": {
            "macro_auroc": best,
            "macro_auprc": float(best_metrics["macro_auprc"]),
            "selection_metric": "Macro AUROC",
            "epoch": best_epoch,
        },
        "epochs": history,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output / "complete.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="ptbxl-superdiagnostic")
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--anchor-lambda", type=float, default=0.1)
    args = parser.parse_args()
    if (args.output / "complete.json").is_file():
        print(f"skip completed {args.output}")
        return

    seed_all(args.seed)
    train_data = HeartLLMFeatureDataset(args.features / "train")
    val_data = HeartLLMFeatureDataset(args.features / "val")
    if train_data.labels.shape[1] != args.classes:
        raise RuntimeError("class count differs from HeartLLM cache")
    args.output.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    campaign_started = time.perf_counter()
    deepsets = train_deepsets(args, train_loader, val_loader, train_data, val_data, device)
    seed_all(args.seed)
    hilar = train_hilar(args, train_loader, val_loader, train_data, val_data, device)
    atomic_json(
        args.output / "complete.json",
        {
            "schema_version": 1,
            "status": "complete",
            "scope": "development train/validation only",
            "formal_test_used": False,
            "experiment": "Second encoder paired DeepSets versus HiLAR",
            "encoder": "HeartLLM ECGEncoder (frozen)",
            "task": args.task,
            "seed": args.seed,
            "features": str(args.features),
            "anchor_lambda": float(args.anchor_lambda),
            "models": {"deepsets": deepsets, "hilar": hilar},
            "elapsed_seconds": time.perf_counter() - campaign_started,
        },
    )


if __name__ == "__main__":
    main()
