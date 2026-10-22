"""CPU FP32 rank-8 geometry with a fused analytic first-order backward."""
import ctypes
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable
from .inference import _cpu_check

NAMES = ('geo_spine', 'geo_t_basis_edge', 'geo_x_basis_edge', 'geo_den_delta',
         'geo_den_weight', 'geo_raw_scale', 'geo_raw_shift',
         'geo_edge_raw_scale', 'geo_edge_raw_shift')
SIZES = (48, 384, 256, 64, 64, 1, 1, 1, 1)
_LIB = None


def library():
    global _LIB
    if _LIB is None:
        _cpu_check(False)
        _LIB = ctypes.CDLL(str(Path(__file__).parent / '.build' / 'backward.so'))
        for name, count in [('forward', 3), ('backward', 5)]:
            fn = getattr(_LIB, 'ks2_train_' + name)
            fn.argtypes = [ctypes.c_void_p] * count + [ctypes.c_int] * 4
            fn.restype = None
    return _LIB


class _Geometry(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, threads, *parameters):
        if coords.device.type != 'cpu' or coords.dtype != torch.float32 or coords.shape[-1] != 8:
            raise ValueError('Expected CPU FP32 rank-8 coordinates')
        if not 1 <= threads <= 1024:
            raise ValueError('Invalid thread count')
        spaces = coords.shape[-2]
        for p, size in zip(parameters, SIZES):
            if p.device.type != 'cpu' or p.dtype != torch.float32 or p.numel() != spaces * size:
                raise ValueError('Incompatible geometry parameter')
        x = coords.contiguous()
        packed = torch.cat([p.reshape(spaces, -1) for p in parameters], 1)
        out = torch.empty_like(x)
        n = x.numel() // (spaces * 8)
        library().ks2_train_forward(x.data_ptr(), packed.data_ptr(), out.data_ptr(), n, spaces, threads, 0)
        ctx.save_for_backward(x, packed, *parameters)
        ctx.threads = threads
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, packed, *parameters = ctx.saved_tensors
        grad = grad.contiguous()
        dx, dp = torch.empty_like(x), torch.empty_like(packed)
        spaces = x.shape[-2]
        library().ks2_train_backward(x.data_ptr(), packed.data_ptr(), grad.data_ptr(), dx.data_ptr(), dp.data_ptr(),
                                     x.numel() // (spaces * 8), spaces, ctx.threads, 0)
        gradients = [g.reshape_as(p) for g, p in zip(dp.split(SIZES, 1), parameters)]
        return dx, None, *gradients


def fused_geo_step(self, coords):
    return _Geometry.apply(coords, self._backward_threads, *(getattr(self, n) for n in NAMES))


def fuse_training(cell, threads=8):
    """Keep parameters, state-dict keys and optimizer ownership unchanged."""
    from types import MethodType
    from ..ks2_stabilized import KS2StabilizedCell
    if not isinstance(cell, KS2StabilizedCell):
        raise TypeError('Expected a stabilized KS2 cell')
    if cell.subspace_rank != 8 or cell.geo_steps != 1:
        raise ValueError('Fused training requires rank 8 and one geometry step')
    library()
    cell._backward_threads = threads
    cell.geo_step = MethodType(fused_geo_step, cell)
    cell.forward = MethodType(fused_forward, cell)
    return cell


class _Cell(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, down, up, threads, *parameters):
        if any(t.device.type != 'cpu' or t.dtype != torch.float32 for t in (inputs, down, up, *parameters)):
            raise ValueError('Fused training requires CPU FP32 tensors')
        spaces = parameters[0].shape[0]
        if down.shape != (spaces*8, inputs.shape[-1]) or up.shape[1] != spaces*8:
            raise ValueError('Incompatible projection shapes')
        if not 1 <= threads <= 1024 or any(p.numel() != spaces*z for p,z in zip(parameters,SIZES)):
            raise ValueError('Invalid threads or geometry shapes')
        x = inputs.reshape(-1, inputs.shape[-1])
        coords = torch.nn.functional.linear(x, down)
        packed = torch.cat([p.reshape(spaces, -1) for p in parameters], 1)
        y = torch.empty_like(coords)
        library().ks2_train_forward(coords.data_ptr(), packed.data_ptr(), y.data_ptr(), len(x), spaces, threads, 1)
        result = torch.nn.functional.linear(y, up)
        ctx.save_for_backward(x, down, up, coords, y, packed, *parameters)
        ctx.shape, ctx.threads = inputs.shape, threads
        return result.reshape(*inputs.shape[:-1], up.shape[0])

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, down, up, coords, y, packed, *parameters = ctx.saved_tensors
        grad = grad.reshape(-1, up.shape[0]).contiguous()
        dy = grad @ up
        dup = grad.T @ y
        dc, dp = torch.empty_like(coords), torch.empty_like(packed)
        spaces = packed.shape[0]
        library().ks2_train_backward(coords.data_ptr(), packed.data_ptr(), dy.data_ptr(), dc.data_ptr(), dp.data_ptr(),
                                     len(x), spaces, ctx.threads, 1)
        ddown = dc.T @ x
        dx = (dc @ down).reshape(ctx.shape)
        gradients = [g.reshape_as(p) for g,p in zip(dp.split(SIZES,1),parameters)]
        return dx, ddown, dup, None, *gradients


def fused_forward(self, inputs):
    return _Cell.apply(inputs, self.down.weight, self.up.weight, self._backward_threads,
                       *(getattr(self, n) for n in NAMES))
