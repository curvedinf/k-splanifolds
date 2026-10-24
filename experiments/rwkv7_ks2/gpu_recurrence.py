"""FP32 Triton RWKV recurrence, with an explicit reverse-mode derivative.

One program per batch/head. Matrix rows index value and columns index key.
The state resets to zero for each sequence, exactly as in run.recurrence.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _forward(R, D, K, V, N, A, Y, S, T: tl.constexpr, H: tl.constexpr,
             W: tl.constexpr):
    bh = tl.program_id(0)
    b, h = bh // H, bh % H
    i = tl.arange(0, W)
    state = tl.full((W, W), 0, tl.float32)
    for t in range(T):
        off = ((b * T + t) * H + h) * W + i
        r, d, k = tl.load(R + off), tl.load(D + off), tl.load(K + off)
        v, n, a = tl.load(V + off), tl.load(N + off), tl.load(A + off)
        removal = tl.sum(state * n[None, :], 1)
        state = state * d[None, :] - removal[:, None] * (n * a)[None, :]
        state = state + v[:, None] * k[None, :]
        tl.store(Y + off, tl.sum(state * r[None, :], 1))
        so = (bh * T + t) * W * W + i[:, None] * W + i[None, :]
        tl.store(S + so, state)


@triton.jit
def _backward(R, D, K, V, N, A, S, DY, DR, DD, DK, DV, DN, DA,
              T: tl.constexpr, H: tl.constexpr, W: tl.constexpr):
    bh = tl.program_id(0)
    b, h = bh // H, bh % H
    i = tl.arange(0, W)
    gs = tl.full((W, W), 0, tl.float32)
    for rev in range(T):
        t = T - 1 - rev
        off = ((b * T + t) * H + h) * W + i
        r, d, k = tl.load(R + off), tl.load(D + off), tl.load(K + off)
        v, n, a = tl.load(V + off), tl.load(N + off), tl.load(A + off)
        dy = tl.load(DY + off)
        so = (bh * T + t) * W * W + i[:, None] * W + i[None, :]
        state = tl.load(S + so)
        prev = tl.load(S + so - W * W, mask=t > 0, other=0)
        tl.store(DR + off, tl.sum(state * dy[:, None], 0))
        gs = gs + dy[:, None] * r[None, :]
        tl.store(DD + off, tl.sum(gs * prev, 0))
        tl.store(DK + off, tl.sum(gs * v[:, None], 0))
        tl.store(DV + off, tl.sum(gs * k[None, :], 1))
        removal = tl.sum(prev * n[None, :], 1)
        grem = -tl.sum(gs * (n * a)[None, :], 1)
        gq = -tl.sum(gs * removal[:, None], 0)
        tl.store(DN + off, gq * a + tl.sum(prev * grem[:, None], 0))
        tl.store(DA + off, gq * n)
        gs = gs * d[None, :] + grem[:, None] * n[None, :]


class _Recurrence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        args = tuple(x.contiguous() for x in args)
        b, t, h, w = args[0].shape
        assert w == 32 and all(x.dtype == torch.float32 and x.is_cuda for x in args)
        y = torch.empty_like(args[0])
        states = torch.empty((b * h, t, w, w), device=y.device, dtype=y.dtype)
        _forward[(b * h,)](*args, y, states, t, h, w, num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(*args, states)
        return y

    @staticmethod
    def backward(ctx, dy):
        *args, states = ctx.saved_tensors
        b, t, h, w = args[0].shape
        grads = [torch.empty_like(x) for x in args]
        _backward[(b * h,)](*args, states, dy.contiguous(), *grads, t, h, w,
                            num_warps=4, enable_fp_fusion=False)
        return tuple(grads)


recurrence = _Recurrence.apply
