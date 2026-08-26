"""Development-only zero-init/anchor ablation for the local residual path."""

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


def masked_average(values: torch.Tensor, valid: torch.Tensor, dim: int) -> torch.Tensor:
    weight = valid.to(values.dtype).unsqueeze(-1)
    return (values * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


class DirectResidual(nn.Module):
    def __init__(self, classes: int, zero_init: bool):
        super().__init__()
        self.lead_embedding = nn.Embedding(12, 32)
        self.direct = nn.Sequential(
            nn.Linear(800, 512), nn.GELU(), nn.LayerNorm(512), nn.Linear(512, 768)
        )
        self.encoder = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 256, bias=False), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(256, 128, bias=False)
        )
        self.delta_head = nn.Linear(128, classes)
        if zero_init:
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)

    def forward(self, local: torch.Tensor, valid: torch.Tensor, baseline: torch.Tensor):
        batch, beats, leads, _ = local.shape
        ids = torch.arange(leads, device=local.device)
        identity = self.lead_embedding(ids)[None, None].expand(batch, beats, -1, -1)
        tokens = self.encoder(self.direct(torch.cat([local, identity], dim=-1)))
        beat_values = masked_average(tokens, valid, dim=2)
        beat_valid = valid.any(dim=2)
        summary = masked_average(beat_values, beat_valid, dim=1)
        delta = self.delta_head(summary)
        return baseline + delta, delta


def autocast_context(device: torch.device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    truths, scores, deltas = [], [], []
    for local, valid, baseline, labels in loader:
        with autocast_context(device):
            logits, delta = model(local.to(device, non_blocking=True), valid.to(device, non_blocking=True).bool(), baseline.to(device, non_blocking=True))
        truths.append(labels.numpy())
        scores.append(torch.sigmoid(logits).float().cpu().numpy())
        deltas.append(delta.float().cpu().numpy())
    truth, score, delta = np.concatenate(truths), np.concatenate(scores), np.concatenate(deltas)
    result = macro_metrics(truth, score)
    result["mean_absolute_delta_logit"] = float(np.abs(delta).mean())
    return result, truth, score


def train_variant(name, zero_init, anchor, args, train_loader, val_loader, train_data, val_data, device):
    output = args.output / name
    if (output / "complete.json").is_file():
        return json.loads((output / "complete.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    model = DirectResidual(args.classes, zero_init).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best = -math.inf
    best_epoch = None
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for local, valid, baseline, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                logits, delta = model(local.to(device, non_blocking=True), valid.to(device, non_blocking=True).bool(), baseline.to(device, non_blocking=True))
                loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.to(device, non_blocking=True))
                loss = loss + float(anchor) * delta.square().mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        validation, truth, score = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **validation}
        history.append(row)
        print(json.dumps({"variant": name, **row}), flush=True)
        if validation["macro_auroc"] > best:
            best = validation["macro_auroc"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch}, output / "checkpoint-best.pth")
            np.savez_compressed(output / "val_predictions.npz", y_true=truth, y_score=score)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "scope": "development train/validation only",
        "formal_test_used": False,
        "experiment": "Anchor / zero-init ablation",
        "task": args.task,
        "seed": args.seed,
        "variant": name,
        "zero_initialized_delta_head": bool(zero_init),
        "anchor_lambda": float(anchor),
        "features": str(args.features),
        "baseline_validation": baseline_from_dataset(val_data),
        "best_validation": {"macro_auroc": best, "selection_metric": "Macro AUROC", "epoch": best_epoch},
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "epochs": history,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output / "complete.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if (args.output / "screen-summary.json").is_file():
        print(f"skip completed {args.output}")
        return
    seed_all(args.seed)
    train_data = CachedFeatureDataset(args.features / "train")
    val_data = CachedFeatureDataset(args.features / "val")
    if train_data.labels.shape[1] != args.classes:
        raise RuntimeError("class count differs from cache")
    args.output.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    campaign_started = time.perf_counter()
    variants = (
        ("zero-init-anchor-01", True, 0.1),
        ("random-init-anchor-01", False, 0.1),
        ("zero-init-no-anchor", True, 0.0),
    )
    results = {}
    for name, zero_init, anchor in variants:
        seed_all(args.seed)
        results[name] = train_variant(name, zero_init, anchor, args, train_loader, val_loader, train_data, val_data, device)
    atomic_json(args.output / "screen-summary.json", {
        "schema_version": 1,
        "status": "complete",
        "scope": "development train/validation only",
        "formal_test_used": False,
        "experiment": "Anchor / zero-init ablation",
        "task": args.task,
        "seed": args.seed,
        "baseline_validation": baseline_from_dataset(val_data),
        "models": results,
        "elapsed_seconds": time.perf_counter() - campaign_started,
    })


if __name__ == "__main__":
    main()
