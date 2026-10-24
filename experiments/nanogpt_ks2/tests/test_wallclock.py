"""Check experimental optimizer execution against the unchanged equations."""
import copy
import unittest

import torch

from experiments.nanogpt_ks2.model import GPT
from experiments.nanogpt_ks2.optim import Optimizer
from experiments.nanogpt_ks2.wallclock_optim import GraphOptimizer
from experiments.nanogpt_ks2.wallclock_graph import capture_model
from experiments.nanogpt_ks2.wallclock_cells import to_hermite, MixedKS3
from experiments.nanogpt_ks2.wallclock_attention import local_attention, install_window_attention


@unittest.skipUnless(torch.cuda.is_available(),'GPU required')
class CapturedOptimizer(unittest.TestCase):
    def test_fused_softcap_loss(self):
        from experiments.nanogpt_ks2.wallclock_softmax import softcap_loss
        torch.manual_seed(105)
        for vocab in (113,10037,24581,50257):
            for dtype in (torch.float32,torch.float16):
                x=(4*torch.randn(17,vocab,device='cuda')).to(dtype).requires_grad_()
                targets=torch.randint(0,vocab,(17,4),device='cuda')
                targets[0,:]=targets[0,0].clone() # Duplicate target coefficients add.
                valid=torch.arange(17,device='cuda')%3!=0
                lp=torch.nn.functional.log_softmax(23*torch.sigmoid((x.float()+5)/7.5),-1)
                rows=torch.arange(17,device='cuda')
                ref=-(lp[rows,targets[:,0]]+.5*lp[rows,targets[:,1]]+
                      .125*lp[rows,targets[:,2]]+.25*valid*lp[rows,targets[:,3]]).mean()
                result=softcap_loss(x,targets,valid,.5,.125,.25)
                torch.testing.assert_close(ref,result,atol=6e-6,rtol=1e-6)
                a=torch.autograd.grad(64*ref,x,retain_graph=True)[0]/64
                b=torch.autograd.grad(64*result,x)[0]/64
                self.assertTrue(torch.isfinite(b).all())
                torch.testing.assert_close(a,b,atol=2e-5 if dtype==torch.float16 else 1e-6,
                                           rtol=.002 if dtype==torch.float16 else 3e-5)

    def test_training_head_objective(self):
        torch.use_deterministic_algorithms(True)
        from types import SimpleNamespace
        from experiments.nanogpt_ks2 import run
        from experiments.nanogpt_ks2.wallclock_loss import loss
        class TinyHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed=torch.nn.Embedding(50257,16,device='cuda')
                torch.nn.init.normal_(self.embed.weight,std=.05)
                self.yarn=SimpleNamespace(apply_window=lambda a,b:None)
                self.yarn_paired_head=self.yarn_wide=self.yarn
            def forward(self,tokens,*args):return self.embed(tokens)
            def head_weight(self):return self.embed.weight
        torch.manual_seed(88);model=TinyHead()
        tokens=torch.randint(0,50257,(131,),device='cuda')
        prefix=torch.arange(50257,device='cuda')-1
        for step in (0,30,65,85,95):
            a,lr=run.loss(model,tokens,prefix,step,100,11,[16,16,48])
            b,lr2=loss(model,tokens,prefix,step,100,11,[16,16,48])
            self.assertEqual(lr,lr2)
            torch.testing.assert_close(a,b,atol=0,rtol=0)
            ga=torch.autograd.grad(a,model.embed.weight)[0]
            gb=torch.autograd.grad(b,model.embed.weight)[0]
            torch.testing.assert_close(ga,gb,atol=0,rtol=0)
            fused,_=loss(model,tokens,prefix,step,100,11,[16,16,48],fused_head_loss=True)
            gf=torch.autograd.grad(fused,model.embed.weight)[0]
            torch.testing.assert_close(a,fused,atol=6e-6,rtol=1e-6)
            torch.testing.assert_close(ga,gf,atol=1e-8,rtol=1e-4)
            half,_=loss(model,tokens,prefix,step,100,11,[16,16,48],head_fp16=True)
            gh=torch.autograd.grad(64*half,model.embed.weight)[0]/64
            self.assertTrue(torch.isfinite(gh).all())
            self.assertLess(((gh-ga).norm()/ga.norm()).item(),.002)

    def test_window_attention_forward_backward(self):
        torch.manual_seed(103)
        torch.backends.cuda.matmul.allow_tf32=False
        for length,heads,d,e,window in [(33,3,8,16,16),(257,6,16,16,176),(513,3,8,8,80)]:
            q=torch.randn(2,length,heads,d,device='cuda',requires_grad=True)
            k=torch.randn_like(q,requires_grad=True)
            v=torch.randn(2,length,heads,e*2,device='cuda')[...,::2].detach().requires_grad_()
            docs=torch.arange(length,device='cuda')//23
            pos=torch.arange(length,device='cuda');delta=pos[:,None]-pos[None,:]
            mask=(delta>=0)&(delta<=window)&(docs[:,None]==docs[None,:])
            ref=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),attn_mask=mask,scale=.27).transpose(1,2)
            result=local_attention(q,k,v,docs,window,.27)
            torch.testing.assert_close(result,ref,atol=2e-6,rtol=2e-5)
            grad=torch.randn_like(result)
            a=torch.autograd.grad(ref,(q,k,v),grad,retain_graph=True)
            b=torch.autograd.grad(result,(q,k,v),grad)
            for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=4e-6,rtol=3e-5)

    def test_mixed_rank_capacity_and_gradients(self):
        torch.manual_seed(71)
        for banks,expected in [([(211,1),(16,8)],67576),
                               ([(211,1),(8,16)],67560),
                               ([(210,1),(32,4)],67408),
                               ([(4,1),(42,8)],67412)]:
            cell=MixedKS3(96,banks).cuda()
            self.assertEqual(sum(p.numel() for p in cell.parameters()),expected)
            torch.nn.init.normal_(cell.up.weight,std=.01)
            x=torch.randn(2,17,96,device='cuda',requires_grad=True)
            y=cell(x)
            self.assertEqual(y.shape,x.shape)
            y.square().mean().backward()
            for p in [x,*cell.parameters()]:
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(p.grad.abs().sum().item(),0.)

    def test_basis_polynomial_identity(self):
        torch.manual_seed(17)
        q=torch.randn(21,6,dtype=torch.float64,requires_grad=True)
        t=torch.linspace(-4,4,21,dtype=torch.float64)
        for basis in ('power','power2','chebyshev2'):
            p=to_hermite(q,basis)
            hermite=(p[:,0]*(2*t**3-3*t*t+1)+p[:,1]*(-2*t**3+3*t*t)
                     +p[:,2]*(t**3-2*t*t+t)+p[:,3]*(t**3-t*t))
            z=t/(2 if basis.endswith('2') else 1)
            if basis.startswith('power'):
                expected=q[:,0]+q[:,1]*z+q[:,2]*z*z+q[:,3]*z**3
            else:
                expected=q[:,0]+q[:,1]*z+q[:,2]*(2*z*z-1)+q[:,3]*(4*z**3-3*z)
            torch.testing.assert_close(hermite,expected,atol=2e-12,rtol=2e-12)
            ga=torch.autograd.grad(hermite.sum(),q,retain_graph=True)[0]
            gb=torch.autograd.grad(expected.sum(),q)[0]
            torch.testing.assert_close(ga,gb,atol=2e-12,rtol=2e-12)

    def test_model_windows_and_shared_head_gradients(self):
        self._check_model_graph(False)

    def test_amp_model_windows_and_shared_head_gradients(self):
        self._check_model_graph(True)

    def test_local_attention_model_graph(self):
        self._check_model_graph(False,True)

    def _check_model_graph(self,amp,local=False):
        torch.set_num_threads(2);torch.manual_seed(81)
        torch.backends.cuda.matmul.allow_tf32=False
        a=GPT('ks3',vocab=128,ngram_rows=64).cuda()
        for ff in a.ff.values():torch.nn.init.normal_(ff.up.weight,std=.01)
        b=copy.deepcopy(a)
        if local:install_window_attention(b)
        capture_model(b,32,amp=amp)
        if amp:
            original=a.forward
            def forward(*args):
                with torch.autocast('cuda',dtype=torch.float16,cache_enabled=False):
                    output=original(*args)
                return output.float()
            a.forward=forward
        old=(16,16,48)
        for window in ((16,48),(48,112),(80,176)):
            new=(window[0],window[0],window[1])
            if new!=old:
                for model in (a,b):
                    for yarn,x,y in zip((model.yarn,model.yarn_paired_head,model.yarn_wide),old,new):
                        yarn.apply_window(x,y)
            old=new
            ids=torch.randint(0,128,(32,),device='cuda')
            docs=torch.arange(32,device='cuda')//11
            ya,yb=a(ids,*window,docs),b(ids,*window,docs)
            torch.testing.assert_close(ya,yb,atol=3e-6 if local else 0,rtol=3e-5 if local else 0)
            # The embedding/head share parameters with captured computation.
            loss_a=torch.nn.functional.linear(ya,a.head_weight()).square().mean()
            loss_b=torch.nn.functional.linear(yb,b.head_weight()).square().mean()
            ga=torch.autograd.grad(loss_a,tuple(a.parameters()),allow_unused=True)
            gb=torch.autograd.grad(loss_b,tuple(b.parameters()),allow_unused=True)
            for x,y in zip(ga,gb):
                self.assertEqual(x is None,y is None)
                if x is not None:torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-5)
        self.assertEqual(b._whole_graph_stats['replays'],3)
        a.eval();b.eval()
        with torch.no_grad():
            torch.testing.assert_close(a(ids,80,176,docs),b(ids,80,176,docs),atol=3e-6 if local else 0,rtol=3e-5 if local else 0)

    def test_schedule_and_moments(self):
        self._check_schedule(False,False)

    def test_batched_schedule_and_moments(self):
        self._check_schedule(True,False)
        self._check_schedule(True,True)

    def test_scaled_gradients(self):
        self._check_schedule(True,True,64.)
        self._check_schedule(True,True,64.,True)

    def _check_schedule(self,batch_matrices,fuse_adam,gradient_scale=1.,scale_in_update=False):
        torch.set_num_threads(2);torch.manual_seed(91)
        torch.backends.cuda.matmul.allow_tf32=False
        a=GPT('ks3',vocab=128,ngram_rows=64).cuda()
        b=copy.deepcopy(a)
        eager=Optimizer(a,.25,fuse_ks3_geometry=False)
        graphed=GraphOptimizer(Optimizer(b,.25,fuse_ks3_geometry=False),
                               batch_matrices=batch_matrices,fuse_adam=fuse_adam,
                               gradient_scale=gradient_scale,scale_in_update=scale_in_update)
        for pa,pb in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(pa,pb,atol=0,rtol=0)
        for progress in (0.,.01,.1,.2,.429,.431,.6,.95,.999):
            for pa,pb in zip(a.parameters(),b.parameters()):
                pa.grad=torch.randn_like(pa)*.01;pb.grad=pa.grad.clone()*gradient_scale
            lr=1-.7*progress
            eager.step(progress,lr);graphed.step(progress,lr)
            for pa,pb in zip(a.parameters(),b.parameters()):
                torch.testing.assert_close(pa,pb,atol=3e-6,rtol=3e-5)
            for sa,sb in zip(eager.matrices,graphed.matrices):
                for key in ('fast','slow','energy'):
                    torch.testing.assert_close(sa[key],sb[key],atol=3e-6,rtol=3e-5)
            for sa,sb in zip(eager.adam,graphed.adam):
                for key in ('m','v'):
                    torch.testing.assert_close(sa[key],sb[key],atol=3e-6,rtol=3e-5)
        self.assertEqual(graphed.fallbacks,0)
        next(a.parameters()).grad=None;next(b.parameters()).grad=None
        if gradient_scale!=1. and not scale_in_update:
            for p in b.parameters():
                if p.grad is not None:p.grad.mul_(gradient_scale)
        eager.step(.999,1.);graphed.step(.999,1.)
        self.assertEqual(graphed.fallbacks,1)
        for pa,pb in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(pa,pb,atol=3e-6,rtol=3e-5)


if __name__=='__main__':unittest.main()
