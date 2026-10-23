"""GPU VJP, optimizer and TP checks for KS3 W8A8 training.

Run unit checks with ``python -m unittest experiments.common.test_ks3_int8``.
Run TP checks with ``torchrun --standalone --nproc-per-node=4 -m
experiments.common.test_ks3_int8 --tp``.
"""
import os
import sys
import unittest

import torch
import torch.distributed as dist
import triton as tr
from torch.nn import functional as F

from .ks3 import KS3
from .ks3_int8 import KS3Int8, _gemm, quantize, pack_weight


def unpack(q, scales, group):
    return q.float() * scales.repeat_interleave(group, -1)[..., :q.shape[-1]]


def straight_through(value, rounded):
    return value + (rounded - value).detach()


def reference(model, x):
    """Independent autograd VJP of the documented smooth, detached-scale STE.

    The real fused forward supplies the quantized feature values, while the
    surrogate derivative uses explicit Hermite/Chebyshev basis evaluation.
    This tests the backward separately from approximation error in the forward.
    """
    down, up, coeff = model.master_down, model.master_up, model.master_coefficients
    qx, sx = quantize(x.detach().reshape(-1, x.shape[-1]), 128)
    qd, sd = pack_weight(down, 128)
    qu, su = pack_weight(up, 92)
    with torch.no_grad():
        ps = coeff.abs().amax(-1).clamp_min(1e-10) / 127
        pc = (coeff / ps[:, None]).round().clamp(-127, 127).to(torch.int8)
        scale = F.softplus(model.master_raw_scale) + 1e-4
        shift = .5 * torch.tanh(model.master_raw_shift)
        m, n = qx.shape[0], down.shape[0]
        h = torch.empty((m, n), device=x.device, dtype=torch.int8)
        hs = torch.empty((m, n // 92), device=x.device)
        _gemm[(tr.cdiv(m, 16), n // 92)](
            qx, sx, qd, sd, h, hs, pc, ps, scale, shift, model.tanh_lut,
            m, n, x.shape[-1], 128, 16, 128, 128, True, model.nonlinear, 0,
            num_warps=4)
    qinput = straight_through(x, unpack(qx, sx, 128).reshape_as(x))
    qdown = straight_through(down, unpack(qd, sd, 128))
    qup = straight_through(up, unpack(qu, su, 92))
    qc = straight_through(coeff, pc.float() * ps[:, None])
    coordinate = F.linear(qinput, qdown).clamp(-65504, 65504)
    t = 4 * torch.tanh(coordinate / 4)
    scale = F.softplus(model.master_raw_scale) + 1e-4
    shift = .5 * torch.tanh(model.master_raw_shift)
    z = torch.tanh(scale * (t + shift))
    basis = torch.stack((2*t**3-3*t*t+1, -2*t**3+3*t*t,
                         t**3-2*t*t+t, t**3-t*t, 2*z*z-1, 4*z**3-3*z), -1)
    features = F.relu(coordinate + (basis * qc).sum(-1)).square()
    features = straight_through(features, unpack(h, hs, 92).reshape_as(features))
    return F.linear(features, qup)


def current_cell(model):
    cell = KS3(model.master_down.shape[1], model.master_up.shape[0], device='cuda')
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
    base = KS3(96).cuda()
    for mode in ('fp32', 'fp16', 'int8'):
        whole = KS3Int8(base, nonlinear=mode, trainable=True)
        shard = KS3Int8(base, rank=rank, world=world, nonlinear=mode, trainable=True)
        x = (torch.randn(2, 17, 96, device='cuda') * .4).requires_grad_()
        a, b = whole(x), shard(x)
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        g = torch.randn_like(a) / a.numel()
        ga = torch.autograd.grad(a, (x, *whole.parameters()), g)
        gb = torch.autograd.grad(b, (x, *shard.parameters()), g)
        torch.testing.assert_close(ga[0], gb[0], atol=2e-6, rtol=5e-4)
        start, stop = rank * (368 // world), (rank+1) * (368 // world)
        for i, (left, right) in enumerate(zip(ga[1:], gb[1:])):
            expected = left[:, start:stop] if i == 1 else left[start:stop]
            torch.testing.assert_close(expected, right, atol=2e-6, rtol=5e-4)
    dist.barrier()
    if rank == 0:
        print(f'PASS: TP{world} forward, input and all parameter gradients; all nonlinear modes')
    dist.destroy_process_group()


if __name__ == '__main__':
    if '--tp' in sys.argv:
        check_tp()
    else:
        unittest.main()
