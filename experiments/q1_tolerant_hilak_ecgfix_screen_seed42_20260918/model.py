"""TolerantECG-compatible keep-one-lead HILA residual adapter."""
from __future__ import annotations

import torch
from torch import nn

from protocol import (EMBED_DIM, GATE_INITIAL_VALUE, HIDDEN_DIM,
                      LEAD_EMBED_DIM, LEADS)


class HILAK(nn.Module):
    def __init__(self, classes: int, baseline_state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.baseline = nn.Linear(EMBED_DIM, classes)
        self.baseline.load_state_dict(baseline_state, strict=True)
        self.baseline.requires_grad_(False)
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
                lead_features: torch.Tensor) -> torch.Tensor:
        if lead_features.ndim != 3 or lead_features.shape[1:] != (LEADS, EMBED_DIM):
            raise ValueError(f"expected (batch,{LEADS},{EMBED_DIM}), got {lead_features.shape}")
        lead_ids = torch.arange(LEADS, device=lead_features.device)
        identity = self.lead_embedding(lead_ids).unsqueeze(0).expand(len(lead_features), -1, -1)
        pooled = self.phi(torch.cat((lead_features, identity), dim=-1)).mean(dim=1)
        correction = self.residual_head(self.rho(pooled))
        return self.baseline(global_features) + self.alpha * correction

