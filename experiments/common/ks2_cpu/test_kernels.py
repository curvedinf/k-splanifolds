import unittest

import torch

from ..ks2 import KS2Cell
from ..ks2_stabilized import KS2Stabilized, KS2StabilizedCell
from . import KS2Horner, KS2VNNI
from .inference import _cpu_check


class StabilizedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)

    def test_shapes_and_backprop(self):
        for model, shape in [(KS2Stabilized(3, 5), (7, 3)),
                             (KS2StabilizedCell(17, 9, num_subspaces=3), (2, 7, 17))]:
            x = torch.randn(*shape, requires_grad=True)
            y = model(x)
            self.assertEqual(y.shape, (*shape[:-1], 5 if shape[-1] == 3 else 9))
            y.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            for p in model.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())

    def test_extreme_inputs_and_gradients(self):
        model = KS2StabilizedCell(8, num_subspaces=2)
        for magnitude in (0, 1e-35, 1e-20, 1, 60000, 1e30):
            for pattern in (torch.ones(3, 8), torch.tensor([1., -1.]).repeat(3, 4)):
                model.zero_grad(set_to_none=True)
                x = (pattern * magnitude).requires_grad_()
                y = model(x)
                y.mean().backward()
                self.assertTrue(torch.isfinite(y).all(), magnitude)
                self.assertTrue(torch.isfinite(x.grad).all(), magnitude)
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()), magnitude)

    def test_calibration_and_autocast(self):
        model = KS2StabilizedCell(8, num_subspaces=2)
        with self.assertRaises(ValueError):
            model.calibrate([])
        x = torch.randn(64, 8)
        model.calibrate([x[:32], x[32:]])
        self.assertTrue(all((getattr(model, 'range_' + n) > 0).all() for n in ('basis', 'atten', 'ex')))
        with torch.autocast('cpu', dtype=torch.bfloat16):
            terms = model.geometry_terms(torch.randn(5, 2, 8))
            self.assertTrue(all(t.dtype == torch.float32 for t in terms))
            self.assertTrue(torch.isfinite(model(x)).all())


class KernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            _cpu_check(True)
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc))
        torch.manual_seed(23)
        torch.set_num_threads(2)
        cls.model = KS2StabilizedCell(17, num_subspaces=3).eval()
        cls.model.calibrate([torch.randn(512, 17)])
        cls.exports = [KS2Horner(cls.model, threads=2), KS2VNNI(cls.model, threads=2)]

    @torch.no_grad()
    def test_export_quality_and_tails(self):
        for n in (0, 1, 15, 16, 17, 127, 128, 129, 512):
            x = torch.randn(n, 17)
            for model in self.exports:
                y = model(x)
                self.assertEqual(y.shape, x.shape)
                self.assertTrue(torch.isfinite(y).all())
                if n:
                    self.assertLess(float((y - self.model(x)).square().mean().sqrt()), .005)
                    singles = torch.cat([model(row[None]) for row in x])
                    torch.testing.assert_close(y, singles, atol=1e-3, rtol=1e-3)

    @torch.no_grad()
    def test_frozen_snapshot_and_noncontiguous_input(self):
        x = torch.randn(17, 12).T
        for model in self.exports:
            before = model(x)
            torch.testing.assert_close(before, model(x.contiguous()))
            fresh = type(model)(self.model.state_dict(), threads=2)
            torch.testing.assert_close(before, fresh(x))
        snapshot = self.exports[0](x)
        original = self.model.down.weight.clone()
        self.model.down.weight.add_(1)
        torch.testing.assert_close(snapshot, self.exports[0](x))
        self.model.down.weight.copy_(original)

    @torch.no_grad()
    def test_stress(self):
        for magnitude in (0, 1e-35, 1e-20, 1, 60000, 1e30):
            x = torch.tensor([1., -1.]).repeat(17, 9)[:, :17] * magnitude
            for model in self.exports:
                self.assertTrue(torch.isfinite(model(x)).all(), magnitude)

    def test_contract_errors(self):
        with self.assertRaises(ValueError):
            KS2Horner(KS2Cell(17))
        with self.assertRaises(ValueError):
            KS2Horner(KS2StabilizedCell(17, geo_steps=2))
        with self.assertRaises(ValueError):
            KS2VNNI(KS2StabilizedCell(17))
        for model in self.exports:
            with self.assertRaises(RuntimeError):
                model(torch.randn(1, 17))
            with self.assertRaises(RuntimeError):
                model.train()
            with self.assertRaises(ValueError):
                model.half()
            with torch.no_grad(), self.assertRaises(ValueError):
                model(torch.randn(1, 2, 17))


if __name__ == '__main__':
    unittest.main()
