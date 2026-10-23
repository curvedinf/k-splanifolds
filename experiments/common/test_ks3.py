"""Check KS3 against direct Hermite/Chebyshev basis evaluation."""

import unittest

import torch
from torch.nn import functional as F

from experiments.common.ks3 import KS3


def basis_reference(cell, inputs):
    x = F.linear(inputs, cell.down.weight).clamp(-65504.0, 65504.0)
    t = 4 * torch.tanh(x / 4)
    z = torch.tanh((F.softplus(cell.raw_scale) + 1e-4) *
                   (t + 0.5 * torch.tanh(cell.raw_shift)))
    basis = torch.stack((
        2 * t**3 - 3 * t**2 + 1,
        -2 * t**3 + 3 * t**2,
        t**3 - 2 * t**2 + t,
        t**3 - t**2,
        2 * z**2 - 1,
        4 * z**3 - 3 * z,
    ), dim=-1)
    geometry = (basis * cell.coefficients).sum(-1)
    return F.linear(F.relu(x + geometry).square(), cell.up.weight)


class TestKS3(unittest.TestCase):
    def test_forward_and_all_gradients_against_basis(self):
        torch.manual_seed(7)
        cell = KS3(9, 5, num_subspaces=13, dtype=torch.float64)
        for amplitude in (0.1, 1.0, 10.0):
            x = (torch.randn(2, 3, 9, dtype=torch.float64) * amplitude).requires_grad_()
            actual, expected = cell(x), basis_reference(cell, x)
            torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
            upstream = torch.randn_like(actual)
            targets = (x, *cell.parameters())
            actual_grads = torch.autograd.grad(actual, targets, upstream)
            expected_grads = torch.autograd.grad(expected, targets, upstream)
            for left, right in zip(actual_grads, expected_grads):
                torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-10)

    def test_winning_budget_and_arbitrary_leading_dimensions(self):
        cell = KS3(96)
        self.assertEqual(sum(p.numel() for p in cell.parameters()), 73600)
        for shape in ((96,), (2, 96), (2, 3, 96), (0, 96)):
            self.assertEqual(cell(torch.randn(shape)).shape, shape)

    def test_low_precision_models_have_finite_gradients(self):
        for dtype in (torch.float16, torch.bfloat16):
            cell = KS3(9, 5, num_subspaces=13, dtype=dtype)
            x = torch.randn(2, 3, 9, dtype=dtype, requires_grad=True)
            y = cell(x)
            self.assertEqual(y.dtype, dtype)
            y.float().square().mean().backward()
            for tensor in (x, *cell.parameters()):
                self.assertTrue(torch.isfinite(tensor.grad).all())


if __name__ == "__main__":
    unittest.main()
