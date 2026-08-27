"""Development-only warm-start attention control.

The original Set-Transformer run trained a new classifier from scratch.  That
is a useful exploratory result, but it is not a fair test of whether attention
adds a local correction to the frozen CLEAR classifier: a from-scratch head
can simply lose the baseline representation.  This variant keeps the frozen
baseline logits and learns a zero-initialized attention residual for the same
ten-epoch budget and seeds.
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
from torch.utils.data import DataLoader

from common import CachedFeatureDataset, atomic_json, baseline_from_dataset, macro_metrics, seed_all


class SetTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        safe = valid.clone()
        empty = ~safe.any(dim=1)
        if empty.any():
            safe[empty, 0] = True
            values = values.clone()
            values[empty, 0] = 0
        normalized = self.norm_attn(values)
        attended, _ = self.attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~safe,
            need_weights=False,
        )
        values = values + attended
        values = values + self.ffn(self.norm_ffn(values))
        return values * valid.unsqueeze(-1).to(values.dtype)


class WarmStartAttentionResidual(nn.Module):
    """Attention over each heartbeat's leads, initialized at baseline logits."""

    def __init__(self, classes: int, input_dim: int = 768, dim: int = 192, heads: int = 4):
        super().__init__()
        self.lead_embedding = nn.Embedding(12, dim)
        self.project = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim), nn.GELU())
        self.blocks = nn.ModuleList([SetTransformerBlock(dim, heads, 0.1) for _ in range(2)])
        self.seed = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.seed, std=dim**-0.5)
        self.pool = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.1)
        self.residual_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, classes))
        # At initialization the attention branch is exactly a no-op.
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self,
        local: torch.Tensor,
        valid: torch.Tensor,
        baseline_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, beats, leads, _ = local.shape
        values = self.project(local)
        ids = torch.arange(leads, device=local.device)
        values = values + self.lead_embedding(ids)[None, None]
        values = values.reshape(batch * beats, leads, -1)
        flat_valid = valid.reshape(batch * beats, leads).bool()
        for block in self.blocks:
            values = block(values, flat_valid)
        safe = flat_valid.clone()
        empty = ~safe.any(dim=1)
        if empty.any():
            safe[empty, 0] = True
            values = values.clone()
            values[empty, 0] = 0
        queries = self.seed.expand(batch * beats, -1, -1)
        pooled, _ = self.pool(queries, values, values, key_padding_mask=~safe, need_weights=False)
        pooled = pooled[:, 0].reshape(batch, beats, -1)
        beat_valid = valid.any(dim=2)
        weight = beat_valid.to(pooled.dtype).unsqueeze(-1)
        summary = (pooled * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        delta = self.residual_head(summary)
        return baseline_logits + delta, delta


def autocast_context(device: torch.device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    truths, scores, deltas = [], [], []
    for local, valid, baseline, labels in loader:
        local = local.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True).bool()
        baseline = baseline.to(device, non_blocking=True)
        with autocast_context(device):
            logits, delta = model(local, valid, baseline)
        truths.append(labels.numpy())
        scores.append(torch.sigmoid(logits).float().cpu().numpy())
        deltas.append(delta.float().cpu().numpy())
    truth = np.concatenate(truths)
    score = np.concatenate(scores)
    delta = np.concatenate(deltas)
    metrics = macro_metrics(truth, score)
    metrics["mean_absolute_delta_logit"] = float(np.abs(delta).mean())
    return metrics, truth, score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--classes", type=int, required=True)
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
    train_data = CachedFeatureDataset(args.features / "train")
    val_data = CachedFeatureDataset(args.features / "val")
    if train_data.labels.shape[1] != args.classes:
        raise RuntimeError("class count differs from cache")
    if not np.isfinite(np.asarray(train_data.logits)).all() or not np.isfinite(np.asarray(val_data.logits)).all():
        raise RuntimeError("baseline logits contain non-finite values")
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
    model = WarmStartAttentionResidual(args.classes).to(device)
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
        for local, valid, baseline, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            local = local.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True).bool()
            baseline = baseline.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast_context(device):
                logits, delta = model(local, valid, baseline)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
                loss = loss + float(args.anchor_lambda) * delta.square().mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        validation, truth, score = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **validation}
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["macro_auroc"] > best:
            best = validation["macro_auroc"]
            best_epoch = epoch
            best_metrics = dict(validation)
            torch.save({"model": model.state_dict(), "epoch": epoch}, args.output / "checkpoint-best.pth")
            np.savez_compressed(args.output / "val_predictions.npz", y_true=truth, y_score=score)

    atomic_json(
        args.output / "complete.json",
        {
            "schema_version": 1,
            "status": "complete",
            "scope": "development train/validation only",
            "formal_test_used": False,
            "experiment": "Warm-start Set Transformer attention residual",
            "task": args.task,
            "seed": args.seed,
            "features": str(args.features),
            "warm_start": "frozen CLEAR baseline logits plus zero-initialized attention residual",
            "anchor_lambda": float(args.anchor_lambda),
            "baseline_validation": baseline_from_dataset(val_data),
            "best_validation": {**best_metrics, "selection_metric": "Macro AUROC", "epoch": best_epoch},
            "epochs": history,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
