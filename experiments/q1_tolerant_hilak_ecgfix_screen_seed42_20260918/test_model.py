"""Architecture invariants for the HILA-K residual path."""
import unittest

import torch
from torch import nn

from model import HILAK
from protocol import EMBED_DIM, LEADS


class HILAKTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        base = nn.Linear(EMBED_DIM, 5)
        self.state = {key: value.detach().clone() for key, value in base.state_dict().items()}
        self.global_features = torch.randn(4, EMBED_DIM)
        self.lead_features = torch.randn(4, LEADS, EMBED_DIM)

    def test_initial_logits_exactly_equal_frozen_baseline(self):
        model = HILAK(5, self.state)
        expected = nn.functional.linear(
            self.global_features, self.state["weight"], self.state["bias"])
        torch.testing.assert_close(
            model(self.global_features, self.lead_features), expected, rtol=0, atol=0)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.baseline.parameters()))

    def test_zero_head_receives_gradient(self):
        model = HILAK(5, self.state)
        labels = torch.randint(0, 2, (4, 5), dtype=torch.float32)
        nn.BCEWithLogitsLoss()(model(self.global_features, self.lead_features), labels).backward()
        self.assertGreater(float(model.residual_head.weight.grad.abs().sum()), 0.0)
        self.assertIsNone(model.baseline.weight.grad)


if __name__ == "__main__":
    unittest.main()

