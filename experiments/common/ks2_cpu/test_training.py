"""Reference VJP, nonzero projection gradients, clamp and tiny-energy coverage."""
import copy
import unittest
import torch
from experiments.nanogpt_ks2.model import FastKS2
from .training import fuse_training, NAMES


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        torch.set_num_threads(8)

    def test_geometry_vjp(self):
        for scale in (0., 1e-20, 1e-15, .01, 1., 10., 1e5):
            with self.subTest(scale=scale):
                ref = FastKS2(96,352)
                native = fuse_training(copy.deepcopy(ref))
                x = (torch.randn(19,7,8)*scale).requires_grad_()
                y = x.detach().clone().requires_grad_()
                upstream = torch.randn_like(x)
                a,b=ref.geo_step(x),native.geo_step(y)
                torch.testing.assert_close(b,a,atol=5e-5,rtol=2e-4)
                (a*upstream).sum().backward();(b*upstream).sum().backward()
                torch.testing.assert_close(y.grad,x.grad,atol=3e-4,rtol=1e-3)
                for name in NAMES:
                    ga,gb=getattr(ref,name).grad,getattr(native,name).grad
                    self.assertTrue(torch.isfinite(gb).all(),name)
                    torch.testing.assert_close(gb,ga,atol=2e-4,rtol=2e-3,msg=lambda m:name+' '+m)

    def test_full_cell_vjp(self):
        ref=FastKS2(96,352)
        torch.nn.init.normal_(ref.up.weight,std=.05)
        native=fuse_training(copy.deepcopy(ref))
        x=torch.randn(2,17,96,requires_grad=True)
        y=x.detach().clone().requires_grad_()
        a,b=ref(x),native(y)
        u=torch.randn_like(a)
        torch.testing.assert_close(b,a,atol=2e-5,rtol=1e-4)
        (a*u).sum().backward();(b*u).sum().backward()
        torch.testing.assert_close(y.grad,x.grad,atol=2e-5,rtol=1e-4)
        for (name,p),(_,q) in zip(ref.named_parameters(),native.named_parameters()):
            torch.testing.assert_close(q.grad,p.grad,atol=5e-5,rtol=1e-3,msg=lambda m:name+' '+m)

    def test_optimizer_trajectory(self):
        ref=FastKS2(96,352)
        torch.nn.init.normal_(ref.up.weight,std=.05)
        native=fuse_training(copy.deepcopy(ref))
        opts=[torch.optim.Adam(m.parameters(),lr=1e-3) for m in (ref,native)]
        for step in range(20):
            x=torch.randn(32,96); target=torch.randn_like(x)
            losses=[]
            for m,opt in zip((ref,native),opts):
                opt.zero_grad(set_to_none=True)
                loss=(m(x)-target).square().mean();loss.backward();opt.step()
                losses.append(loss.detach())
            torch.testing.assert_close(losses[0],losses[1],atol=1e-5,rtol=1e-5)
        for p,q in zip(ref.parameters(),native.parameters()):
            torch.testing.assert_close(p,q,atol=2e-4,rtol=2e-3)


if __name__=='__main__':unittest.main()
