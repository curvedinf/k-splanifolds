"""Mixed precision, compilation, checkpointing and external-weight integration."""
import unittest

import torch
from torch.utils.checkpoint import checkpoint

from experiments.common.ks3 import KS3
from experiments.common.ks3_int8 import capture_training, fuse_training, ks3_int8


@unittest.skipUnless(torch.cuda.is_available(), 'GPU and Triton required')
class Compatibility(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False

    def test_parameter_ownership_and_state_dict(self):
        cell=KS3(9,13,7,3).cuda()
        parameters=dict(cell.named_parameters())
        keys=set(cell.state_dict())
        fuse_training(cell)
        self.assertEqual(set(cell.state_dict()),keys)
        for name,p in cell.named_parameters():
            self.assertIs(p,parameters[name])
        restored=fuse_training(KS3(9,13,7,3).cuda())
        restored.load_state_dict(cell.state_dict())
        x=torch.randn(2,5,9,device='cuda')*.2
        torch.testing.assert_close(cell(x),restored(x),atol=0,rtol=0)

    def test_input_dtypes_and_autocast(self):
        for dtype in (torch.float16,torch.bfloat16):
            for autocast in (False,True):
                cell=fuse_training(KS3(9,13,7,3).cuda())
                x=(torch.randn(2,5,9,device='cuda')*.2).to(torch.float32 if autocast else dtype).requires_grad_()
                with torch.autocast('cuda',dtype=dtype,enabled=autocast):
                    actual=cell(x)
                expected=cell(x.float()).to(dtype)
                self.assertEqual(actual.dtype,dtype)
                torch.testing.assert_close(actual,expected,atol=0,rtol=0)
                grad=torch.randn_like(actual)
                targets=(x,*cell.parameters())
                ga=torch.autograd.grad(actual,targets,grad)
                gb=torch.autograd.grad(expected,targets,grad)
                for a,b in zip(ga,gb):
                    torch.testing.assert_close(a,b,atol=2e-6,rtol=5e-4)
                self.assertEqual(ga[0].dtype,x.dtype)
                self.assertTrue(all(g.dtype==torch.float32 for g in ga[1:]))

    def test_compiled_forward_backward(self):
        for rank in (1,3):
            cell=fuse_training(KS3(9,13,7,rank).cuda())
            eager_forward=cell.forward
            names=list(dict(cell.named_parameters()))
            cell.compile(fullgraph=True,dynamic=False)
            self.assertEqual(list(dict(cell.named_parameters())),names)
            x=(torch.randn(2,5,9,device='cuda')*.2).requires_grad_()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                a,b=cell(x),eager_forward(x)
            torch.testing.assert_close(a,b,atol=0,rtol=0)
            g=torch.randn_like(a);targets=(x,*cell.parameters())
            ga=torch.autograd.grad(a,targets,g)
            gb=torch.autograd.grad(b,targets,g)
            for left,right in zip(ga,gb):
                torch.testing.assert_close(left,right,atol=2e-6,rtol=5e-4)

    def test_activation_checkpoint(self):
        cell=fuse_training(KS3(9,13,7,3).cuda())
        x=(torch.randn(2,5,9,device='cuda')*.2).requires_grad_()
        a=checkpoint(cell,x,use_reentrant=False)
        b=cell(x)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        targets=(x,*cell.parameters());g=torch.randn_like(a)
        for l,r in zip(torch.autograd.grad(a,targets,g),torch.autograd.grad(b,targets,g)):
            torch.testing.assert_close(l,r,atol=0,rtol=0)

    def test_parameter_bank_views(self):
        bank=torch.nn.Parameter(torch.randn(2,21,9,device='cuda')*.1)
        coeff=torch.nn.Parameter(torch.randn(7,3,6,device='cuda')*.05)
        warp=torch.nn.Parameter(torch.stack((torch.full((7,),.75,device='cuda'),
                                            torch.zeros(7,device='cuda')),dim=-1))
        x=torch.randn(2,5,9,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        y=ks3_int8(x,bank[0],bank[1].T,coeff,warp[:,0],warp[:,1],subspace_rank=3)
        expected=ks3_int8(x,bank[0],bank[1].T,coeff,warp[:,0].contiguous(),warp[:,1].contiguous(),subspace_rank=3)
        torch.testing.assert_close(y,expected,atol=0,rtol=0)
        targets=(x,bank,coeff,warp);g=torch.randn_like(y)
        for a,b in zip(torch.autograd.grad(y,targets,g),torch.autograd.grad(expected,targets,g)):
            self.assertTrue(torch.isfinite(a).all())
            self.assertGreater(a.abs().sum().item(),0)
            torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_graph_replay_updates_and_shape_fallback(self):
        eager=fuse_training(KS3(9,13,7,3).cuda())
        captured=fuse_training(KS3(9,13,7,3).cuda())
        captured.load_state_dict(eager.state_dict())
        x=(torch.randn(1,16,9,device='cuda')*.2).requires_grad_()
        identities={name:id(p) for name,p in captured.named_parameters()}
        capture_training(captured,x)
        self.assertEqual(identities,{name:id(p) for name,p in captured.named_parameters()})
        self.assertEqual(set(eager.state_dict()),set(captured.state_dict()))
        # Reuse the graph after parameter updates; other shapes fall back.
        for shape in ((1,16,9),(1,16,9),(2,5,9),(1,16,9)):
            x=(torch.randn(shape,device='cuda')*.2).requires_grad_()
            a,b=eager(x),captured(x)
            torch.testing.assert_close(a,b,atol=0,rtol=0)
            g=torch.randn_like(a)
            ga=torch.autograd.grad(a,(x,*eager.parameters()),g)
            gb=torch.autograd.grad(b,(x,*captured.parameters()),g)
            for left,right in zip(ga,gb):
                torch.testing.assert_close(left,right,atol=0,rtol=0)
            with torch.no_grad():
                for pa,pb,gradient in zip(eager.parameters(),captured.parameters(),ga[1:]):
                    pa.add_(gradient,alpha=-1e-5)
                    pb.add_(gradient,alpha=-1e-5)
        eager.eval();captured.eval()
        with torch.no_grad():
            torch.testing.assert_close(eager(x),captured(x),atol=0,rtol=0)


if __name__=='__main__':
    unittest.main()
