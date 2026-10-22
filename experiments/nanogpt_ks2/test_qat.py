"""Check packed-forward equivalence and the intended straight-through derivative."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from experiments.nanogpt_ks2.train_qat import Ternary
from experiments.nanogpt_ks2.tq1_gs128 import roundtrip


def test_ternary():
    torch.manual_seed(17)
    for w in (torch.zeros(3,7),torch.randn(7,432),torch.randn(96,56),torch.tensor([[0.,.5,-.5,1.,-1.]])):
        w.requires_grad_();q=Ternary.apply(w);packed,_=roundtrip(w)
        torch.testing.assert_close(q,packed,rtol=0,atol=0)
        grad=torch.randn_like(q);q.backward(grad)
        torch.testing.assert_close(w.grad,grad,rtol=0,atol=0)


if __name__=='__main__':
    torch.set_num_threads(1)
    test_ternary()
    print('Packed forward and STE checks passed')
