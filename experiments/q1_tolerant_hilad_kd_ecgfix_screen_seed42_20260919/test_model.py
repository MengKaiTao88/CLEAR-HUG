from __future__ import annotations

import unittest

import torch

from model import HILA
from protocol import EMBED_DIM, LEADS, VARIANTS


class HILATest(unittest.TestCase):
    def setUp(self) -> None:
        self.classes = 5
        self.state = {"weight": torch.randn(self.classes, EMBED_DIM),
                      "bias": torch.randn(self.classes)}
        self.global_features = torch.randn(3, EMBED_DIM)
        self.keep = torch.randn(3, LEADS, EMBED_DIM)
        self.delta = torch.randn(3, LEADS, EMBED_DIM)

    def test_initial_logits_equal_frozen_baseline(self) -> None:
        expected = torch.nn.functional.linear(
            self.global_features, self.state["weight"], self.state["bias"])
        for variant in VARIANTS:
            model = HILA(self.classes, self.state, variant)
            actual = model(self.global_features, self.keep, self.delta)
            self.assertTrue(torch.equal(expected, actual), variant)

    def test_residual_head_receives_gradient(self) -> None:
        for variant in VARIANTS:
            model = HILA(self.classes, self.state, variant)
            model(self.global_features, self.keep, self.delta).sum().backward()
            self.assertGreater(float(model.residual_head.weight.grad.abs().sum()), 0.0, variant)
            self.assertIsNone(model.baseline.weight.grad)


if __name__ == "__main__":
    unittest.main()
