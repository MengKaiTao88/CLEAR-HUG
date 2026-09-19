from __future__ import annotations

import unittest
import torch

from model import HILAK, HILAKLRA
from protocol import EMBED_DIM, LEADS, MAX_BEATS


class LRATest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2)
        self.classes = 4
        self.baseline = {"weight": torch.randn(self.classes, EMBED_DIM),
                         "bias": torch.randn(self.classes)}
        hilak = HILAK(self.classes, self.baseline)
        self.hilak_state = hilak.state_dict()
        self.global_f = torch.randn(3, EMBED_DIM)
        self.lead_f = torch.randn(3, LEADS, EMBED_DIM)
        self.local_f = torch.randn(3, MAX_BEATS, EMBED_DIM)
        self.mask = torch.zeros(3, MAX_BEATS, dtype=torch.bool); self.mask[:, :8] = True

    def test_initial_logits_exactly_equal_frozen_hilak(self) -> None:
        expected = HILAK(self.classes, self.baseline)
        expected.load_state_dict(self.hilak_state)
        model = HILAKLRA(self.classes, self.baseline, self.hilak_state)
        self.assertTrue(torch.equal(expected(self.global_f, self.lead_f),
                                    model(self.global_f, self.lead_f, self.local_f, self.mask)))

    def test_only_lra_is_trainable_and_head_receives_gradient(self) -> None:
        model = HILAKLRA(self.classes, self.baseline, self.hilak_state)
        model(self.global_f, self.lead_f, self.local_f, self.mask).sum().backward()
        self.assertGreater(float(model.local_head.weight.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.hilak.parameters()))


if __name__ == "__main__":
    unittest.main()
