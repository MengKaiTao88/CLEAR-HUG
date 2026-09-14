#!/usr/bin/env python3
"""Official ST-MEM downstream fine-tuning adapted only for multilabel targets."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Subset

from common import array_sha256, atomic_json, metrics, sha256, state_dict
from official_protocol import (
    ACTUAL_LR, BASE_LR, BATCH_SIZE, CAMPAIGN, EPOCHS, FRACTIONS,
    SEEDS, SELECTION_METRIC, STMEM_SHA256, TASKS, WARMUP_EPOCHS,
    WEIGHT_DECAY,
)


class ArrayDataset(Dataset):
    def __init__(self, directory: Path, split: str, transform):
        self.x = np.load(directory / f"{split}_data.npy", mmap_mode="r")
        self.y = np.load(directory / f"{split}_labels.npy", mmap_mode="r")
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        value = np.asarray(self.x[index], dtype=np.float32)
        if value.shape == (1000, 12):
            value = value.T
        return self.transform(value), torch.as_tensor(np.asarray(self.y[index]), dtype=torch.float32)


def transforms_for(repo: Path, split: str):
    sys.path.insert(0, str(repo))
    import util.transforms as transforms
    common = [
        transforms.HighpassFilter(fs=250, cutoff=0.67),
        transforms.LowpassFilter(fs=250, cutoff=40),
        transforms.Standardize(axis=(-1, -2)),
    ]
    if split == "train":
        operations = [transforms.RandomCrop(crop_length=2250), *common]
        operations.append(transforms.RandAugment(
            ops=transforms.get_transforms_from_config({name: {} for name in (
                "shift", "cutout", "drop", "flip", "erase", "sine",
                "partial_sine", "partial_white_noise",
            )}), level=10, num_layers=2, prob=0.5,
        ))
    else:
        operations = [transforms.NCrop(crop_length=2250, num_segments=3), *common]
    operations.append(transforms.ToTensor())
    pipeline = transforms.Compose(operations)
    resample = transforms.Resample(target_fs=250)

    def apply(value):
        return pipeline(resample(value, 100))
    return apply


def load_model(root: Path, classes: int):
    repo = root / "external_models/ST-MEM"
    sys.path.insert(0, str(repo))
    from models.encoder.st_mem_vit import st_mem_vit_base
    model = st_mem_vit_base(num_leads=12, num_classes=classes, seq_len=2250, patch_size=75)
    checkpoint = root / "model_weights/ST-MEM/st_mem_vit_base_encoder.pth"
    if sha256(checkpoint) != STMEM_SHA256:
        raise RuntimeError("official ST-MEM checkpoint SHA256 mismatch")
    payload = {key.removeprefix("module."): value for key, value in state_dict(torch.load(checkpoint, map_location="cpu")).items()}
    for key in ("head.weight", "head.bias"):
        if key in payload and payload[key].shape != model.state_dict()[key].shape:
            del payload[key]
    incompatible = model.load_state_dict(payload, strict=False)
    if set(incompatible.missing_keys) != {"head.weight", "head.bias"} or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")
    return model, checkpoint


def lr_at(epoch: float) -> float:
    if epoch < WARMUP_EPOCHS:
        return ACTUAL_LR * epoch / WARMUP_EPOCHS
    return ACTUAL_LR * 0.5 * (1 + math.cos(math.pi * (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)))


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval(); truths = []; scores = []; total_loss = 0.0; total = 0
    for samples, targets in loader:
        samples = samples.cuda(non_blocking=True); targets = targets.cuda(non_blocking=True)
        with torch.cuda.amp.autocast():
            if samples.ndim == 4:
                logits_list = torch.stack([model(samples[:, crop]) for crop in range(samples.shape[1])], dim=1)
                logits = logits_list.mean(dim=1)
                probabilities = torch.sigmoid(logits_list).mean(dim=1)
            else:
                logits = model(samples); probabilities = torch.sigmoid(logits)
            loss = criterion(logits, targets)
        total_loss += float(loss) * len(targets); total += len(targets)
        truths.append(targets.cpu().numpy()); scores.append(probabilities.float().cpu().numpy())
    truth = np.concatenate(truths); score = np.concatenate(scores)
    return total_loss / total, metrics(truth, score)


def save_checkpoint(path: Path, model, epoch: int, validation: dict):
    incoming = path.with_name(path.name + ".incoming")
    torch.save({"model": model.state_dict(), "epoch": epoch, "validation": validation,
                "protocol": "official-stmem-downstream-finetune"}, incoming)
    os.replace(incoming, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", choices=tuple(FRACTIONS), required=True)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    complete = args.output / "training-complete.json"
    if complete.exists(): return
    if args.output.exists(): raise RuntimeError(f"refusing incomplete output {args.output}")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    classes, _, raw_rel, total_records = TASKS[args.task]
    raw = args.root / "campaign-inputs" / __import__("official_protocol").SOURCE_CAMPAIGN / "raw" / raw_rel
    repo = args.root / "external_models/ST-MEM"
    train_dataset = ArrayDataset(raw, "train", transforms_for(repo, "train"))
    val_dataset = ArrayDataset(raw, "val", transforms_for(repo, "val"))
    if len(train_dataset) != total_records: raise RuntimeError("training record count mismatch")
    count = total_records if FRACTIONS[args.fraction] == 1 else int(total_records * FRACTIONS[args.fraction])
    indices = np.arange(total_records, dtype=np.int64) if count == total_records else np.asarray(random.Random(args.seed).sample(range(total_records), count), dtype=np.int64)
    train_dataset = Subset(train_dataset, indices.tolist())
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset, generator=generator), batch_size=BATCH_SIZE, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)
    model, checkpoint = load_model(args.root, classes); model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=ACTUAL_LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.BCEWithLogitsLoss(); scaler = torch.cuda.amp.GradScaler()
    args.output.mkdir(parents=True)
    atomic_json(args.output / "subset-manifest.json", {"fraction": args.fraction, "task": args.task, "seed": args.seed, "selected_records": count, "indices_sha256": array_sha256(indices)})
    best_loss = float("inf"); best_auc = -float("inf"); best_loss_epoch = None; best_auc_epoch = None
    log = args.output / "log.txt"
    for epoch in range(EPOCHS):
        model.train(); losses = []
        for step, (samples, targets) in enumerate(train_loader):
            lr = lr_at(epoch + step / len(train_loader))
            for group in optimizer.param_groups: group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(): loss = criterion(model(samples.cuda(non_blocking=True)), targets.cuda(non_blocking=True))
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); losses.append(float(loss))
        val_loss, validation = evaluate(model, val_loader, criterion)
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "lr": lr, "val_loss": val_loss, **validation}
        with log.open("a") as handle: handle.write(json.dumps(row) + "\n")
        if val_loss < best_loss:
            best_loss = val_loss; best_loss_epoch = epoch + 1
            save_checkpoint(args.output / "checkpoint-best-loss.pth", model, epoch + 1, row)
        if validation["macro_auroc"] > best_auc:
            best_auc = validation["macro_auroc"]; best_auc_epoch = epoch + 1
            save_checkpoint(args.output / "checkpoint-best-AUROC.pth", model, epoch + 1, row)
        print(json.dumps(row), flush=True)
    payload = {"status": "complete", "campaign": CAMPAIGN, "model": "stmem", "mode": "finetune", "task": args.task, "fraction_key": args.fraction, "seed": args.seed,
               "epochs": EPOCHS, "batch_size": BATCH_SIZE, "base_lr": BASE_LR, "actual_lr": ACTUAL_LR, "warmup_epochs": WARMUP_EPOCHS, "weight_decay": WEIGHT_DECAY,
               "random_crop": 2250, "randaugment": True, "eval_crops": 3, "selection_metric": SELECTION_METRIC, "test_used_for_selection": False,
               "best_validation_loss": best_loss, "best_loss_epoch": best_loss_epoch, "best_validation_auroc": best_auc, "best_auroc_epoch": best_auc_epoch,
               "checkpoint_sha256": sha256(args.output / "checkpoint-best-loss.pth"), "encoder_sha256": sha256(checkpoint), "subset_indices_sha256": array_sha256(indices)}
    atomic_json(complete, payload); print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
