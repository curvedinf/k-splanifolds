"""GPU VJP, optimizer and TP checks for KS3 W8A8 training.

Run unit checks with ``python -m unittest experiments.common.test_ks3_int8``.
Run TP checks with ``torchrun --standalone --nproc-per-node=4 -m
experiments.common.test_ks3_int8 --tp``.
"""
import os
import sys
import unittest
from types import SimpleNamespace

import torch
import torch.distributed as dist
import triton as tr
from torch.nn import functional as F

from .ks3 import KS3
from .ks3_int8 import (KS3Int8, Int8Linear, Int8Controller, Int8Embedding,
                      quantize, pack_weight, pack_spine, _project_features, chart_range)


def unpack(q, scales, group):
    return q.float() * scales.float().repeat_interleave(group, -1)[..., :q.shape[-1]]


def straight_through(value, rounded):
    return value + (rounded - value).detach()


def reference(model, x):
    """Independent autograd VJP of the documented smooth, detached-scale STE.

    The real fused forward supplies the quantized feature values, while the
    surrogate derivative uses explicit Hermite/Chebyshev basis evaluation.
    This tests the backward separately from approximation error in the forward.
    """
    down, up, coeff = model.master_down, model.master_up, model.master_coefficients
    qx, sx = quantize(x.detach().reshape(-1, x.shape[-1]), 32)
    qd, sd = pack_weight(down, 32)
    qu, su = pack_weight(up, 32)
    with torch.no_grad():
        pc,ps=pack_spine(coeff)
        scale=F.softplus(model.master_raw_scale)+1e-4
        shift=.5*torch.tanh(model.master_raw_shift)
        h,hs,_,_=_project_features(qx,sx,qd,sd,pc,ps,scale,shift,model.tanh_lut,
                                    model.subspace_rank,model.nonlinear)
    qinput = straight_through(x, unpack(qx, sx, 32).reshape_as(x))
    qdown = straight_through(down, unpack(qd, sd, 32))
    qup = straight_through(up, unpack(qu, su, 32))
    qc = straight_through(coeff, (pc.float() * ps.float()[:, None]).reshape_as(coeff))
    coordinate = F.linear(qinput, qdown).clamp(-65504, 65504)
    coordinate=coordinate.reshape(*x.shape[:-1],model.local_subspaces,model.subspace_rank)
    t = 4 * torch.tanh(coordinate.mean(-1,keepdim=True) / 4)
    scale = F.softplus(model.master_raw_scale)[:,None] + 1e-4
    shift = .5 * torch.tanh(model.master_raw_shift)[:,None]
    z = torch.tanh(scale * (t + shift))
    basis = torch.stack((2*t**3-3*t*t+1, -2*t**3+3*t*t,
                         t**3-2*t*t+t, t**3-t*t, 2*z*z-1, 4*z**3-3*z), -1)
    features = F.relu(coordinate + (basis * qc.reshape(model.local_subspaces,model.subspace_rank,6)).sum(-1)).square()
    features = straight_through(features, unpack(h, hs, 32).reshape_as(features))
    return F.linear(features.flatten(-2), qup)


def current_cell(model):
    cell = KS3(model.in_features,model.out_features,model.num_subspaces,model.subspace_rank,device='cuda')
    with torch.no_grad():
        for dest, source in ((cell.down.weight, model.master_down),
                             (cell.up.weight, model.master_up),
                             (cell.coefficients, model.master_coefficients),
                             (cell.raw_scale, model.master_raw_scale),
                             (cell.raw_shift, model.master_raw_shift)):
            dest.copy_(source)
    return cell


@unittest.skipUnless(torch.cuda.is_available(), 'GPU and Triton required')
class TestKS3Int8(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False

    def test_gs32_half_scales_and_partial_groups(self):
        values=torch.randn(4,67,device='cuda')*torch.tensor([0.,1e-9,1e3,1e8],device='cuda')[:,None]
        q,s=quantize(values)
        packed,scales=pack_weight(values)
        self.assertEqual(s.dtype,torch.float16)
        self.assertEqual(s.shape,(4,3))
        self.assertTrue(torch.isfinite(s).all() and (s>0).all())
        torch.testing.assert_close(q,packed,atol=0,rtol=0)
        torch.testing.assert_close(s,scales,atol=0,rtol=0)
        weight=torch.randn(13,67,device='cuda')*.1
        linear=Int8Linear(weight)
        expected=unpack(q,s,32)@unpack(linear.qweight,linear.scales,32).T
        torch.testing.assert_close(linear(values),expected,atol=2.,rtol=3e-5)
        embedding=Int8Embedding(weight)
        ids=torch.tensor([[0,3],[7,12]],device='cuda')
        torch.testing.assert_close(embedding(ids),unpack(embedding.qweight,embedding.scales,32)[ids],atol=0,rtol=0)
        self.assertEqual(embedding.scales.dtype,torch.float16)

    def test_controller_partial_groups(self):
        for activation in ('linear','tanh','sigmoid'):
            source=SimpleNamespace(down=torch.nn.Linear(67,65,bias=False).cuda(),
                                   up=torch.nn.Linear(65,19,bias=False).cuda(),activation=activation)
            controller=Int8Controller(source)
            x=torch.randn(17,67,device='cuda')*.2
            h=controller.down(x)
            if activation=='tanh':h=torch.tanh(h)
            elif activation=='sigmoid':h=torch.sigmoid(h)
            expected=controller.up(h)
            torch.testing.assert_close(controller(x),expected,atol=2e-6,rtol=2e-4)

    def test_variable_dimensions_counts_and_ranks(self):
        for d,o,s,r in ((1,1,1,1),(7,13,5,3),(129,65,37,2),(33,17,3,16),(5,9,2,65)):
          for mode in ('fp32','fp16','int8'):
            with self.subTest(shape=(d,o,s,r),mode=mode):
                base=KS3(d,o,s,r).cuda()
                model=KS3Int8(base,nonlinear=mode,trainable=True)
                x=(torch.randn(3,d,device='cuda')*.3).requires_grad_()
                actual,expected=model(x),reference(model,x)
                torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
                with torch.no_grad():
                    torch.testing.assert_close(actual,KS3Int8(base,nonlinear=mode)(x),atol=0,rtol=0)
                g=torch.randn_like(actual)/actual.numel();targets=(x,*model.parameters())
                for l,v in zip(torch.autograd.grad(actual,targets,g),torch.autograd.grad(expected,targets,g)):
                    torch.testing.assert_close(l,v,atol=2e-6,rtol=5e-4)
                for name,value in model.named_buffers():
                    if name.endswith(('scales','spine_scale')):
                        self.assertEqual(value.dtype,torch.float16,name)
                empty=torch.empty(0,d,device='cuda',requires_grad=True)
                y=model(empty)
                self.assertEqual(y.shape,(0,o))
                y.sum().backward()
                self.assertEqual(empty.grad.shape,empty.shape)

    def test_forward_and_all_ste_gradients(self):
        for mode in ('fp32', 'fp16', 'int8'):
            for shape in ((1, 96), (2, 17, 96)):
                with self.subTest(mode=mode, shape=shape):
                    base = KS3(96).cuda()
                    model = KS3Int8(base, nonlinear=mode, trainable=True)
                    x = (torch.randn(*shape, device='cuda') * .4).requires_grad_()
                    actual = model(x)
                    with torch.no_grad():
                        inference = KS3Int8(base, nonlinear=mode)(x)
                    torch.testing.assert_close(actual, inference, atol=0, rtol=0)
                    expected = reference(model, x)
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                    grad = torch.randn_like(actual) / actual.numel()
                    targets = (x, *model.parameters())
                    ga = torch.autograd.grad(actual, targets, grad)
                    ge = torch.autograd.grad(expected, targets, grad)
                    for name, left, right in zip(('input', *dict(model.named_parameters())), ga, ge):
                        torch.testing.assert_close(left, right, atol=2e-6, rtol=5e-4,
                                                   msg=lambda message: name + ': ' + message)

    def test_optimizer_repacking_and_accumulation(self):
        model = KS3Int8(KS3(96).cuda(), trainable=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
        x = torch.randn(32, 96, device='cuda') * .4
        losses = []
        for _ in range(12):
            optimizer.zero_grad(set_to_none=True)
            loss = model(x).square().mean()
            losses.append(loss.item())
            loss.backward()
            for parameter in model.parameters():
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
            optimizer.step()
        self.assertLess(losses[-1], losses[0])
        with torch.no_grad():
            expected = KS3Int8(current_cell(model))(x)
            torch.testing.assert_close(model(x), expected, atol=0, rtol=0)
            exported = model.to_inference()
            self.assertEqual(list(exported.parameters()), [])
            torch.testing.assert_close(exported(x), expected, atol=0, rtol=0)
            restored = KS3Int8(KS3(96).cuda(), trainable=True)
            restored.load_state_dict(model.state_dict())
            torch.testing.assert_close(restored(x), model(x), atol=0, rtol=0)
        # Two outstanding forward graphs must retain independent packed operands.
        optimizer.zero_grad(set_to_none=True)
        first, second = model(x), model(x * .5)
        (first.square().mean() + second.square().mean()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_noncontiguous_input_and_clamp(self):
        model = KS3Int8(KS3(96).cuda(), nonlinear='fp32', trainable=True)
        for amplitude in (0., .2, 1e6):
            x = (torch.randn(96, 17, device='cuda') * amplitude).T.requires_grad_()
            a, b = model(x), reference(model, x)
            g = torch.randn_like(a) / max(amplitude, 1.)**2
            targets = (x, *model.parameters())
            ga = torch.autograd.grad(a, targets, g)
            gb = torch.autograd.grad(b, targets, g)
            for left, right in zip(ga, gb):
                torch.testing.assert_close(left, right, atol=1e-4, rtol=1e-3)


def check_tp():
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    dist.init_process_group('nccl', device_id=torch.device('cuda', rank))
    torch.manual_seed(83)
    for d,o,charts,chart_rank in ((96,96,368,1),(7,13,65,3),(9,5,3,16),(1,1,1,1)):
      base=KS3(d,o,charts,chart_rank).cuda()
      for mode in ('fp32','fp16','int8'):
        whole=KS3Int8(base,nonlinear=mode,trainable=True)
        shard=KS3Int8(base,rank=rank,world=world,nonlinear=mode,trainable=True)
        x=(torch.randn(2,17,d,device='cuda')*.4).requires_grad_()
        a,b=whole(x),shard(x)
        torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
        g=torch.randn_like(a)/a.numel()
        ga=torch.autograd.grad(a,(x,*whole.parameters()),g)
        gb=torch.autograd.grad(b,(x,*shard.parameters()),g)
        torch.testing.assert_close(ga[0],gb[0],atol=2e-6,rtol=5e-4)
        start,stop=chart_range(charts,chart_rank,rank,world)
        for i,(left,right) in enumerate(zip(ga[1:],gb[1:])):
            if i==0:expected=left[start*chart_rank:stop*chart_rank]
            elif i==1:expected=left[:,start*chart_rank:stop*chart_rank]
            else:expected=left[start:stop]
            torch.testing.assert_close(expected,right,atol=2e-6,rtol=5e-4)
    dist.barrier()
    if rank == 0:
        print(f'PASS: TP{world} forward, input and all parameter gradients; all nonlinear modes')
    dist.destroy_process_group()


if __name__ == '__main__':
    if '--tp' in sys.argv:
        check_tp()
    else:
        unittest.main()
