"""Global-only controls with parameter counts matched to HILA-K + LRA."""
from __future__ import annotations

import torch
from torch import nn

from protocol import EMBED_DIM, GATE_INITIAL_VALUE, hilar_parameter_budget


def trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class ParameterMatchedMLP(nn.Module):
    """Standalone two-hidden-layer global MLP classifier."""

    def __init__(self, classes: int) -> None:
        super().__init__()
        target = hilar_parameter_budget(classes)

        def size(width: int) -> int:
            return ((EMBED_DIM + 1) * width + (width + 1) * width
                    + (width + 1) * classes)

        self.width = min(range(1, 6001), key=lambda value: abs(size(value) - target))
        self.network = nn.Sequential(
            nn.Linear(EMBED_DIM, self.width), nn.GELU(),
            nn.Linear(self.width, self.width), nn.GELU(),
            nn.Linear(self.width, classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class GenericResidualAdapter(nn.Module):
    """Unstructured global embedding adapter with a frozen LP anchor."""

    def __init__(self, classes: int, baseline_state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.baseline = nn.Linear(EMBED_DIM, classes)
        self.baseline.load_state_dict(baseline_state, strict=True)
        self.baseline.requires_grad_(False)
        target = hilar_parameter_budget(classes)

        def size(width: int) -> int:
            return ((EMBED_DIM + 1) * width + (width + 1) * EMBED_DIM
                    + (EMBED_DIM + 1) * classes + classes)

        self.width = min(range(1, 6001), key=lambda value: abs(size(value) - target))
        self.adapter = nn.Sequential(
            nn.Linear(EMBED_DIM, self.width), nn.GELU(),
            nn.Linear(self.width, EMBED_DIM), nn.GELU(),
        )
        self.residual_head = nn.Linear(EMBED_DIM, classes)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.alpha = nn.Parameter(torch.full((classes,), GATE_INITIAL_VALUE))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.baseline(features) + self.alpha * self.residual_head(self.adapter(features))
