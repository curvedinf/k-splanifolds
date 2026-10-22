import ctypes
import platform
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ..ks2_stabilized import KS2StabilizedCell, _STRENGTH_OFFSET


def _cpu_check(vnni):
    required = {'avx2', 'fma', 'avx512f', 'avx512bw', 'avx512dq', 'avx512vl'}
    if vnni:
        required.add('avx512_vnni')
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
        raise RuntimeError('Native KS2 requires x86-64 Linux')
    flags = next((set(line.split(':', 1)[1].split()) for line in
                  Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('flags')), set())
    if missing := required - flags:
        raise RuntimeError('Missing CPU features: ' + ', '.join(sorted(missing)))


def _state(source):
    if isinstance(source, nn.Module):
        if isinstance(source, KS2StabilizedCell):
            if source.subspace_rank != 8 or source.geo_steps != 1:
                raise ValueError('Native export requires rank 8 and one geometry step')
        elif (getattr(source, 'r', None), getattr(source, 'h', None),
              getattr(source, 'combine', None), getattr(source, 'mode', None)) != (8, 1, 'sum', 'combined'):
            raise ValueError('Expected KS2StabilizedCell or a packed combined-QAT model')
        source = source.state_dict()
    if not isinstance(source, Mapping):
        raise TypeError('Expected a module, state dict, or checkpoint containing state_dict')
    state = source.get('state_dict', source)
    if 'geo_spine' in state:
        s = state['geo_spine'].shape[0]
        state = dict(state, geo_t=torch.cat((state['geo_spine'], state['geo_t_basis_edge'].reshape(s, 6, -1)), -1),
                     geo_x=state['geo_x_basis_edge'].permute(0, 2, 1, 3))
        for old, new in [('geo_den_delta', 'den_delta'), ('geo_den_weight', 'den_weight'),
                         ('geo_raw_scale', 'raw_scale'), ('geo_raw_shift', 'raw_shift'),
                         ('geo_edge_raw_scale', 'edge_raw_scale'), ('geo_edge_raw_shift', 'edge_raw_shift')]:
            state[new] = state[old]
    keys = ('down.weight', 'up.weight', 'geo_t', 'geo_x', 'den_delta', 'den_weight',
            'raw_scale', 'raw_shift', 'edge_raw_scale', 'edge_raw_shift')
    if missing := set(keys) - state.keys():
        raise ValueError('Missing stabilized weights: ' + ', '.join(sorted(missing)))
    keys += tuple('range_' + n for n in ('basis', 'atten', 'ex') if 'range_' + n in state)
    result = {k: state[k].detach().to(device='cpu', dtype=torch.float32).contiguous().clone() for k in keys}
    if not all(torch.isfinite(v).all() for v in result.values()):
        raise ValueError('Weights and ranges must be finite')
    s = result['geo_t'].shape[0]
    shapes = {'geo_t': (s, 6, 72), 'geo_x': (s, 8, 4, 8),
              'den_delta': (s, 8, 8), 'den_weight': (s, 8, 8)}
    if s < 1 or any(tuple(result[k].shape) != shape for k, shape in shapes.items()):
        raise ValueError('Expected rank-8 packed geometry')
    down, up = result['down.weight'], result['up.weight']
    if down.ndim != 2 or up.ndim != 2 or down.shape[0] != s * 8 or up.shape[1] != s * 8:
        raise ValueError('Projection shapes do not match the geometry')
    for k in ('raw_scale', 'raw_shift', 'edge_raw_scale', 'edge_raw_shift'):
        if result[k].numel() != s:
            raise ValueError('Expected one warp parameter per chart')
    return result


def _quantize(weight):
    s = weight.shape[0]
    flat = weight.reshape(s, -1)
    blocks = F.pad(flat, (0, (-flat.shape[1]) % 128)).reshape(s, -1, 128)
    scale = blocks.abs().amax(-1).clamp_min(1e-8) / 127
    q = (blocks / scale[..., None]).round().clamp(-127, 127).to(torch.int8)
    return q.reshape(s, -1)[:, :flat.shape[1]].contiguous(), scale


def _dequantize(weight):
    q, scale = _quantize(weight)
    return (q.float() * scale.repeat_interleave(128, -1)[:, :q.shape[1]]).reshape_as(weight)


class _FrozenKS2(nn.Module):
    def _buffer(self, name, value):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite exported buffer: ' + name)
        self.register_buffer(name, value.detach().contiguous().clone())

    def _setup(self, state, threads, backend):
        _cpu_check(backend == 'vnni')
        if not isinstance(threads, int) or not 1 <= threads <= 1024:
            raise ValueError('threads must be an integer between 1 and 1024')
        self.spaces, self.threads = state['geo_t'].shape[0], threads
        self._buffer('down', state['down.weight'])
        self._buffer('up', state['up.weight'])
        for name, key in [('aa', 'den_delta'), ('bb', 'den_weight')]:
            self._buffer(name, 8 * torch.sigmoid(state[key] + _STRENGTH_OFFSET))
        self._buffer('warp', torch.stack((F.softplus(state['raw_scale'].flatten()) + 1e-4,
                     .5 * torch.tanh(state['raw_shift'].flatten()),
                     F.softplus(state['edge_raw_scale'].flatten()) + 1e-4,
                     .5 * torch.tanh(state['edge_raw_shift'].flatten())), 1))
        path = Path(__file__).parent / '.build' / (backend + '.so')
        if not path.is_file():
            raise RuntimeError('Build kernels: python -m experiments.common.ks2_cpu.build')
        self._lib = ctypes.CDLL(str(path))
        self._fn = getattr(self._lib, 'ks2_' + backend)
        self._fn.argtypes = [ctypes.c_void_p] * (10 if backend == 'vnni' else 7) + [ctypes.c_int] * 3
        self._fn.restype = None
        self.eval()

    def train(self, mode=True):
        if mode:
            raise RuntimeError('Frozen inference export; train KS2StabilizedCell instead')
        return super().train(False)

    def _apply(self, fn, recurse=True):
        probe = fn(torch.empty(0, dtype=torch.float32))
        if probe.device.type != 'cpu' or probe.dtype != torch.float32:
            raise ValueError('Native buffers must stay on CPU in their exported dtypes')
        return super()._apply(fn, recurse)

    def forward(self, x):
        if torch.is_grad_enabled():
            raise RuntimeError('Use torch.no_grad() or torch.inference_mode()')
        if x.device.type != 'cpu' or x.dtype != torch.float32:
            raise ValueError('Expected CPU float32 input')
        if x.ndim != 2 or x.shape[1] != self.down.shape[1]:
            raise ValueError('Expected [batch, in_features] input')
        if len(x) * self.spaces * 8 >= 2**31:
            raise ValueError('Input exceeds native indexing range')
        c = F.linear(x, self.down).contiguous()
        y = torch.empty_like(c)
        if len(x):
            self._fn(c.data_ptr(), *(getattr(self, k).data_ptr() for k in self._arguments),
                     y.data_ptr(), len(x), self.spaces, self.threads)
        return F.linear(y, self.up)


class KS2Horner(_FrozenKS2):
    """GS128 weight values folded into FP32 coefficients; 16-patch AVX-512 geometry."""
    _arguments = ('a', 'q', 'aa', 'bb', 'warp')

    @torch.no_grad()
    def __init__(self, source, threads=8):
        super().__init__()
        state = _state(source)
        self._setup(state, threads, 'horner')
        a0, a1, a2, a3, a4, a5 = _dequantize(state['geo_t']).unbind(1)
        self._buffer('a', torch.stack((a0, a2, -3*a0 + 3*a1 - 2*a2 - a3,
                                      2*a0 - 2*a1 + a2 + a3, a4, a5), 1))
        q0, q1, q2, q3 = _dequantize(state['geo_x']).unbind(2)
        self._buffer('q', torch.stack((q0 - q2, q1 - 3*q3, 2*q2, 4*q3), 1))


class KS2VNNI(_FrozenKS2):
    """GS128 INT8 coefficients, VNNI dots, INT32 edges and FP32 projections."""
    _arguments = ('wt', 'ts', 'wx', 'xs', 'aa', 'bb', 'warp', 'ranges')

    @torch.no_grad()
    def __init__(self, source, threads=8):
        super().__init__()
        state = _state(source)
        self._setup(state, threads, 'vnni')
        ranges = [state.get('range_' + n) for n in ('basis', 'atten', 'ex')]
        if any(v is None or v.shape != (self.spaces,) or not bool((v > 0).all()) for v in ranges):
            raise ValueError('Positive per-chart ranges required; call model.calibrate(batches) first')
        self._buffer('ranges', torch.stack([v.clamp_min(1e-8) / 127 for v in ranges], 1))
        for name in ('aa', 'bb', 'warp'):
            value = getattr(self, name).half().float()
            if not bool(torch.isfinite(value).all()):
                raise ValueError('Warp exceeds the supported export range')
            setattr(self, name, value)
        wt, ts = _quantize(state['geo_t'])
        wx, xs = _quantize(state['geo_x'])
        packed = torch.zeros(self.spaces, 5, 4, 2, 16, 4, dtype=torch.int8)
        for j in range(72):
            for k in range(6):
                packed[:, j//16, (k*72+j)//128, k//4, j%16, k%4] = wt[:, k*72+j]
        correction = (-128 * packed.int().sum((3, 5))).to(torch.int32)
        self._buffer('wt', torch.cat((packed.flatten(), correction.contiguous().view(torch.int8).flatten())))
        packed = wx.reshape(self.spaces, 8, 4, 8).transpose(2, 3).contiguous()
        correction = (-128 * packed.int().sum(-1)).to(torch.int32)
        self._buffer('wx', torch.cat((packed.flatten(), correction.contiguous().view(torch.int8).flatten())))
        self._buffer('ts', ts)
        self._buffer('xs', xs)
