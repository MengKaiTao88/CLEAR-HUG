"""End-to-end validation-batch efficiency benchmark for CLEAR models.

The benchmark includes host-to-device transfer, the released CLEAR encoder,
lead aggregation, and the direct HiLAR residual path.  It opens only a fixed
batch from the validation arrays and records the device, precision, batch,
latency, throughput, peak VRAM, parameter counts, and best-effort FLOPs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch


def add_paths(root: Path) -> None:
    sys.path[:0] = [str(root / "src/CLEAR-HUG"), str(root / "mvp")]


def load_state(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    return payload.get("model", payload) if isinstance(payload, dict) else payload


def load_fair(name: str, checkpoint: Path, classes: int):
    from timm.models import create_model

    import modeling_finetune_fair_baselines  # noqa: F401

    model = create_model(
        name,
        pretrained=False,
        num_classes=classes,
        cls_token_num=12,
        padding_mask=False,
        atten_mask=True,
        mask_ratio=0,
    )
    incompatible = model.load_state_dict(load_state(checkpoint), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch for {name}: {incompatible}")
    return model


def masked_mean(values: torch.Tensor, valid: torch.Tensor, dim: int):
    from modeling_finetune_fair_baselines import masked_mean as fair_masked_mean

    return fair_masked_mean(values, valid, dim=dim)


def set_policy(encoder, fair, residual=None):
    encoder.requires_grad_(False)
    fair.backbone.requires_grad_(False)
    if residual is not None:
        # The HiLAR row is an adapter on top of a frozen DeepSets baseline.
        fair.adapter.requires_grad_(False)
        fair.mlp_head.requires_grad_(False)
        residual.requires_grad_(False)
        residual.direct.requires_grad_(True)
        residual.encoder.requires_grad_(True)
        residual.delta_head.requires_grad_(True)
    else:
        fair.adapter.requires_grad_(True)
        fair.mlp_head.requires_grad_(True)


def count_parameters(*modules):
    total = sum(p.numel() for module in modules for p in module.parameters())
    trainable = sum(p.numel() for module in modules for p in module.parameters() if p.requires_grad)
    return int(total), int(trainable)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--released-checkpoint", type=Path, required=True)
    parser.add_argument("--hug-checkpoint", type=Path, required=True)
    parser.add_argument("--deepsets-checkpoint", type=Path, required=True)
    parser.add_argument("--hilar-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if "test" in {part.lower() for part in args.dataset.parts}:
        raise ValueError("test paths are forbidden")
    add_paths(args.root)
    from clear_adapter import ClearFeatureAdapter
    from modeling_deepsets_anchored_residual import AnchoredResidualClassifier

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the end-to-end benchmark")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    qrs = np.load(args.dataset / "val_data.npy", mmap_mode="r")[: args.batch_size]
    times = np.load(args.dataset / "val_data_in_times.npy", mmap_mode="r")[: args.batch_size]
    channel_path = args.dataset / "val_data_in_chans.npy"
    if channel_path.is_file():
        channels = np.load(channel_path, mmap_mode="r")[: args.batch_size]
    else:
        channels = np.tile(
            np.repeat(np.arange(1, 13, dtype=np.int64), 15),
            (len(qrs), 1),
        )
    cpu_qrs = torch.from_numpy(np.asarray(qrs, dtype=np.float32))
    cpu_channels = torch.from_numpy(np.asarray(channels, dtype=np.int64))
    cpu_times = torch.from_numpy(np.asarray(times, dtype=np.int64))
    batch = len(cpu_qrs)

    def build(kind: str):
        encoder = ClearFeatureAdapter(args.root / "src/CLEAR-HUG", args.released_checkpoint).to(device).eval()
        if kind == "HUG":
            fair = load_fair("CLEAR_MASKED_HUG_finetune_base", args.hug_checkpoint, args.classes).to(device).eval()
            residual = None
        else:
            fair = load_fair("CLEAR_MASKED_DEEPSETS_finetune_base", args.deepsets_checkpoint, args.classes).to(device).eval()
            residual = None
            if kind == "HiLAR / Final Direct":
                residual = AnchoredResidualClassifier(args.classes, "direct").to(device).eval()
                residual.load_state_dict(load_state(args.hilar_checkpoint), strict=True)
        set_policy(encoder, fair, residual)
        return encoder, fair, residual

    def forward(kind: str, encoder, fair, residual):
        # Include host-to-device transfer so every row is genuinely end-to-end.
        signals = cpu_qrs.to(device, non_blocking=True)
        channels_t = cpu_channels.to(device, non_blocking=True)
        times_t = cpu_times.to(device, non_blocking=True)
        local, lead_cls = encoder(signals, channels_t, times_t)
        valid = times_t.reshape(batch, 12, 15).gt(0).permute(0, 2, 1)
        lead_valid = valid.any(dim=1)
        if kind == "HUG":
            groups, group_valid = fair.adapter(lead_cls, lead_valid)
            summary, _ = masked_mean(groups, group_valid, dim=1)
            return fair.mlp_head(summary)
        baseline_summary = fair.adapter(lead_cls, lead_valid)
        baseline_logits = fair.mlp_head(baseline_summary)
        if kind == "DeepSets":
            return baseline_logits
        zero_shared = torch.zeros(batch, 15, 768, device=device, dtype=local.dtype)
        logits, _ = residual(local, zero_shared, valid, baseline_logits)
        return logits

    rows = []
    for kind in ("HUG", "DeepSets", "HiLAR / Final Direct"):
        encoder, fair, residual = build(kind)
        for _ in range(args.warmup):
            with torch.autocast("cuda", dtype=torch.float16), torch.inference_mode():
                forward(kind, encoder, fair, residual)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        elapsed = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16), torch.inference_mode():
                forward(kind, encoder, fair, residual)
            torch.cuda.synchronize()
            elapsed.append((time.perf_counter() - start) * 1000.0)
        flop_value = None
        flop_error = None
        try:
            with torch.profiler.profile(with_flops=True) as profile:
                with torch.autocast("cuda", dtype=torch.float16), torch.inference_mode():
                    forward(kind, encoder, fair, residual)
                torch.cuda.synchronize()
            flop_value = int(sum(int(event.flops or 0) for event in profile.key_averages()))
        except Exception as error:  # pragma: no cover - profiler varies by build
            flop_error = f"{type(error).__name__}: {error}"
        total, trainable = count_parameters(encoder, fair.adapter, fair.mlp_head, *(tuple([residual]) if residual is not None else tuple()))
        median_ms = statistics.median(elapsed)
        rows.append(
            {
                "model": kind,
                "total_parameters": total,
                "trainable_parameters": trainable,
                "batch_size": batch,
                "qrs_tokens": 180,
                "samples_per_token": 96,
                "precision": "fp16 autocast",
                "latency_ms": {
                    "p50": median_ms,
                    "p95": sorted(elapsed)[max(0, round(0.95 * (len(elapsed) - 1)))],
                    "mean": statistics.fmean(elapsed),
                },
                "throughput_records_per_second": batch * 1000.0 / median_ms,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
                "flops_per_batch": flop_value,
                "flops_profiler_error": flop_error,
            }
        )
        del encoder, fair, residual
        torch.cuda.empty_cache()

    payload = {
        "schema_version": 1,
        "mode": "end_to_end_validation_batch_benchmark",
        "scope": "development validation only",
        "formal_test_used": False,
        "task": "PTB-XL Superdiagnostic",
        "dataset": str(args.dataset),
        "split": "val",
        "records_in_fixed_batch": batch,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "models": rows,
        "notes": [
            "Latency includes host-to-device transfer, frozen CLEAR encoding, aggregation, and residual inference.",
            "The same validation batch, precision, batch size, and QRS token shape are used for every model.",
            "FLOPs are best-effort PyTorch profiler totals and may omit unsupported operators.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    incoming = args.output.with_suffix(args.output.suffix + ".incoming")
    incoming.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    incoming.replace(args.output)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
