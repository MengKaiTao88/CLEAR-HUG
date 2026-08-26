"""Train the validation-only lead-set head on HeartLLM encoder features."""

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


class HeartLLMDataset(Dataset):
    def __init__(self, root: Path):
        self.features = np.load(root / "features.npy", mmap_mode="r")
        self.labels = np.load(root / "labels.npy", mmap_mode="r")
        if len(self.features) != len(self.labels):
            raise RuntimeError("HeartLLM cache arrays have inconsistent lengths")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return np.asarray(self.features[index], dtype=np.float32), np.asarray(self.labels[index], dtype=np.float32)


class LeadSetHead(nn.Module):
    def __init__(self, classes: int, input_dim: int = 32, dim: int = 128, heads: int = 4):
        super().__init__()
        self.lead_embedding = nn.Embedding(12, dim)
        self.project = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim), nn.GELU())
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.1)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, classes))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, leads, _ = features.shape
        ids = torch.arange(leads, device=features.device)
        values = self.project(features) + self.lead_embedding(ids)[None]
        normalized = values
        attended, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        values = values + attended
        values = values + self.ffn(values)
        return self.head(values.mean(dim=1))


def autocast_context(device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval(); truths=[]; scores=[]
    for features, labels in loader:
        with autocast_context(device):
            logits = model(features.to(device, non_blocking=True))
        truths.append(labels.numpy()); scores.append(torch.sigmoid(logits).float().cpu().numpy())
    truth, score = np.concatenate(truths), np.concatenate(scores)
    return macro_metrics(truth, score), truth, score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="ptbxl-superdiagnostic")
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if (args.output / "complete.json").is_file():
        print(f"skip completed {args.output}"); return
    seed_all(args.seed)
    train_data, val_data = HeartLLMDataset(args.features / "train"), HeartLLMDataset(args.features / "val")
    if train_data.labels.shape[1] != args.classes:
        raise RuntimeError("class count differs from HeartLLM cache")
    args.output.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeadSetHead(args.classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best=-math.inf; best_epoch=None; history=[]; started=time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); losses=[]
        for features, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                logits=model(features.to(device, non_blocking=True))
                loss=nn.functional.binary_cross_entropy_with_logits(logits, labels.to(device, non_blocking=True))
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
        validation, truth, score=evaluate(model,val_loader,device)
        row={"epoch":epoch,"train_loss":float(np.mean(losses)),**validation}; history.append(row); print(json.dumps(row),flush=True)
        if validation["macro_auroc"]>best:
            best=validation["macro_auroc"]; best_epoch=epoch
            torch.save({"model":model.state_dict(),"epoch":epoch},args.output/"checkpoint-best.pth")
            np.savez_compressed(args.output/"val_predictions.npz",y_true=truth,y_score=score)
    atomic_json(args.output/"complete.json",{
        "schema_version":1,"status":"complete","scope":"development train/validation only",
        "formal_test_used":False,"experiment":"Second encoder validation",
        "encoder":"HeartLLM ECGEncoder (frozen)","task":args.task,"seed":args.seed,
        "features":str(args.features),"best_validation":{"macro_auroc":best,"selection_metric":"Macro AUROC","epoch":best_epoch},
        "epochs":history,"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
        "elapsed_seconds":time.perf_counter()-started,
    })


if __name__ == "__main__":
    main()
