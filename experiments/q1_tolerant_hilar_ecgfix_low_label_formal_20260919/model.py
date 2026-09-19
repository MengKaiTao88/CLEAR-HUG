"""Locked HILA-K and heartbeat-local residual architectures."""
from __future__ import annotations

import torch
from torch import nn

from protocol import (EMBED_DIM, GATE_INITIAL_VALUE, HILA_HIDDEN_DIM,
                      LEAD_EMBED_DIM, LEADS, LRA_DIMS)


class HILAK(nn.Module):
    def __init__(self, classes: int, baseline_state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.baseline = nn.Linear(EMBED_DIM, classes)
        self.baseline.load_state_dict(baseline_state, strict=True)
        self.baseline.requires_grad_(False)
        self.lead_embedding = nn.Embedding(LEADS, LEAD_EMBED_DIM)
        self.phi = nn.Sequential(
            nn.Linear(EMBED_DIM + LEAD_EMBED_DIM, HILA_HIDDEN_DIM), nn.GELU(),
            nn.Linear(HILA_HIDDEN_DIM, HILA_HIDDEN_DIM), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(HILA_HIDDEN_DIM, EMBED_DIM), nn.GELU())
        self.residual_head = nn.Linear(EMBED_DIM, classes)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.alpha = nn.Parameter(torch.full((classes,), GATE_INITIAL_VALUE))

    def forward(self, global_features: torch.Tensor,
                lead_features: torch.Tensor) -> torch.Tensor:
        lead_ids = torch.arange(LEADS, device=lead_features.device)
        identity = self.lead_embedding(lead_ids).unsqueeze(0).expand(len(lead_features), -1, -1)
        pooled = self.phi(torch.cat((lead_features, identity), dim=-1)).mean(dim=1)
        return self.baseline(global_features) + self.alpha * self.residual_head(self.rho(pooled))


class HILAKLRA(nn.Module):
    def __init__(self, classes: int, baseline_state: dict[str, torch.Tensor],
                 hilak_state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.hilak = HILAK(classes, baseline_state)
        self.hilak.load_state_dict(hilak_state, strict=True)
        self.hilak.requires_grad_(False)
        layers: list[nn.Module] = []
        source = EMBED_DIM
        for target in LRA_DIMS:
            layers.extend((nn.Linear(source, target), nn.GELU()))
            source = target
        self.local_mlp = nn.Sequential(*layers)
        self.local_head = nn.Linear(LRA_DIMS[-1], classes)
        nn.init.zeros_(self.local_head.weight)
        nn.init.zeros_(self.local_head.bias)
        self.local_alpha = nn.Parameter(torch.full((classes,), GATE_INITIAL_VALUE))

    def forward(self, global_features: torch.Tensor, lead_features: torch.Tensor,
                local_features: torch.Tensor, local_mask: torch.Tensor) -> torch.Tensor:
        base_logits = self.hilak(global_features, lead_features)
        tokens = self.local_mlp(local_features)
        weights = local_mask.to(tokens.dtype).unsqueeze(-1)
        pooled = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return base_logits + self.local_alpha * self.local_head(pooled)
