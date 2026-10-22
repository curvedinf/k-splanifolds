"""Parity, causality, precision and control checks. Run with python -m unittest."""
import unittest
import torch
from .model import GPT, FastKS2
from experiments.common.ks2_stabilized import KS2StabilizedCell


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_horner_forward_and_backward(self):
        torch.manual_seed(7)
        fast = FastKS2(96, 352)
        ref = KS2StabilizedCell(96, 96, 7, 8, 1)
        torch.nn.init.normal_(fast.up.weight, std=.05)
        ref.load_state_dict(fast.state_dict())
        x = torch.randn(2, 9, 96, requires_grad=True)
        z = x.detach().clone().requires_grad_(True)
        a, b = fast(x), ref(z)
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        a.square().sum().backward(); b.square().sum().backward()
        torch.testing.assert_close(x.grad, z.grad, atol=2e-5, rtol=1e-4)
        for p, q in zip(fast.parameters(), ref.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-4)

    def test_shared_initialization_and_budget(self):
        torch.manual_seed(11); a = GPT('mlp', vocab=128)
        torch.manual_seed(11); b = GPT('ks2', vocab=128)
        for name, p in a.state_dict().items():
            if not name.startswith('ff.'):
                self.assertTrue(torch.equal(p, b.state_dict()[name]), name)
        ratio = sum(p.numel() for p in b.ff.parameters())/sum(p.numel() for p in a.ff.parameters())
        self.assertTrue(.24 < ratio <= .25)
        x = torch.randint(0, 128, (32,)); doc = torch.zeros_like(x)
        torch.testing.assert_close(a(x,16,32,doc), b(x,16,32,doc), rtol=0, atol=0)
        self.assertTrue(all(p.dtype == torch.float32 for p in b.parameters()))

    def test_causality_after_nonzero_ffn(self):
        torch.manual_seed(1); m = GPT('ks2', vocab=128)
        for ff in m.ff.values(): torch.nn.init.normal_(ff.up.weight,std=.03)
        x = torch.randint(0,128,(32,)); changed=x.clone(); changed[16:] = torch.randint(0,128,(16,))
        doc = torch.zeros_like(x)
        torch.testing.assert_close(m(x,16,32,doc)[:16],m(changed,16,32,doc)[:16], rtol=1e-5,atol=1e-6)

    def test_extreme_geometry_stays_finite(self):
        m = FastKS2(96,352)
        x = torch.randn(4,7,8)*1e8
        self.assertTrue(torch.isfinite(m.geo_step(x)).all())


if __name__ == '__main__':unittest.main()
