"""TolerantECG-compatible contextual HILA-D and fused HILA-KD adapters."""
from __future__ import annotations

import torch
from torch import nn

from protocol import (EMBED_DIM, GATE_INITIAL_VALUE, HIDDEN_DIM,
                      LEAD_EMBED_DIM, LEADS, VARIANTS)


class HILA(nn.Module):
    def __init__(self, classes: int, baseline_state: dict[str, torch.Tensor],
                 variant: str) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        self.variant = variant
        self.baseline = nn.Linear(EMBED_DIM, classes)
        self.baseline.load_state_dict(baseline_state, strict=True)
        self.baseline.requires_grad_(False)
        self.fusion = (nn.Sequential(nn.Linear(2 * EMBED_DIM, EMBED_DIM), nn.GELU())
                       if variant == "HILA-KD" else None)
        self.lead_embedding = nn.Embedding(LEADS, LEAD_EMBED_DIM)
        self.phi = nn.Sequential(
            nn.Linear(EMBED_DIM + LEAD_EMBED_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
        )
        self.rho = nn.Sequential(nn.Linear(HIDDEN_DIM, EMBED_DIM), nn.GELU())
        self.residual_head = nn.Linear(EMBED_DIM, classes)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.alpha = nn.Parameter(torch.full((classes,), GATE_INITIAL_VALUE))

    def forward(self, global_features: torch.Tensor,
                keep_features: torch.Tensor,
                delta_features: torch.Tensor) -> torch.Tensor:
        expected = (LEADS, EMBED_DIM)
        if keep_features.ndim != 3 or keep_features.shape[1:] != expected:
            raise ValueError(f"expected keep (batch,{LEADS},{EMBED_DIM}), got {keep_features.shape}")
        if delta_features.ndim != 3 or delta_features.shape[1:] != expected:
            raise ValueError(f"expected delta (batch,{LEADS},{EMBED_DIM}), got {delta_features.shape}")
        leads = (delta_features if self.variant == "HILA-D" else
                 self.fusion(torch.cat((keep_features, delta_features), dim=-1)))
        lead_ids = torch.arange(LEADS, device=leads.device)
        identity = self.lead_embedding(lead_ids).unsqueeze(0).expand(len(leads), -1, -1)
        pooled = self.phi(torch.cat((leads, identity), dim=-1)).mean(dim=1)
        correction = self.residual_head(self.rho(pooled))
        return self.baseline(global_features) + self.alpha * correction
