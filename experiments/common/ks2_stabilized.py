"""Bounded KS2: smooth cubic cap, energy damping, FP32 geometry."""

import math
from functools import wraps

import torch
from torch.nn import functional as F

from .ks2 import KS2, KS2Cell

_STRENGTH_OFFSET = math.log(math.log1p(math.exp(-4)) / (8 - math.log1p(math.exp(-4)))) + 4


def _fp32_geometry(fn):
    @wraps(fn)
    def wrapped(self, coords):
        with torch.autocast(device_type=coords.device.type, enabled=False):
            return fn(self, coords)
    return wrapped


def _coordinates(coords):
    return coords.float().clamp(-65504, 65504)


def _predicates(x):
    energy = x.square()
    total = energy.sum(-1, keepdim=True)
    deviation = (x - x.mean(-1, keepdim=True)).square()
    importance = energy / total.clamp_min(1e-30)
    distance = deviation / (deviation + total / x.shape[-1]).clamp_min(1e-30)
    return distance, importance


def _strength(raw):
    return 8 * torch.sigmoid(raw.float() + _STRENGTH_OFFSET)


class KS2Stabilized(KS2):
    """KS2 map with bounded geometry; parameter layout matches KS2."""

    @_fp32_geometry
    def forward(self, coords):
        x = _coordinates(coords)
        t = 4 * torch.tanh(x.mean(-1, keepdim=True) / 4)
        basis = self.spine_basis(t)
        spine = torch.einsum('...m,mo->...o', basis, self.geo_spine.float())
        edge_t = torch.einsum('...m,mio->...io', basis, self.geo_t_basis_edge.float())
        edge_x = torch.einsum('...ik,kio->...io', self.edge_basis(x), self.geo_x_basis_edge.float())
        distance, importance = _predicates(x)
        den = 1 + _strength(self.geo_den_delta) * distance[..., None]
        den = den + _strength(self.geo_den_weight) * importance[..., None]
        return (spine + (edge_t * edge_x / den).sum(-2)).to(coords.dtype)

    @staticmethod
    def warp(x, raw_scale, raw_shift):
        return KS2.warp(x.float(), raw_scale.float(), raw_shift.float())


class KS2StabilizedCell(KS2Cell):
    """Trainable stabilized cell. Native exports support rank 8 and one step."""

    def __init__(self, in_features, out_features=None, num_subspaces=4,
                 subspace_rank=8, geo_steps=1):
        if min(in_features, in_features if out_features is None else out_features,
               num_subspaces, subspace_rank, geo_steps) < 1:
            raise ValueError('Dimensions and geo_steps must be positive')
        super().__init__(in_features, out_features, num_subspaces, subspace_rank, geo_steps)
        for name in ('basis', 'atten', 'ex'):
            self.register_buffer('range_' + name, torch.zeros(num_subspaces))

    @_fp32_geometry
    def geometry_terms(self, coords):
        x = _coordinates(coords)
        t = 4 * torch.tanh(x.mean(-1, keepdim=True) / 4)
        basis = self.spine_basis(t)
        spine = torch.einsum('nsm,smr->nsr', basis, self.geo_spine.float())
        edge_t = torch.einsum('nsm,smro->nsro', basis, self.geo_t_basis_edge.float())
        edge_x = torch.einsum('nsrk,skro->nsro', self.edge_basis(x), self.geo_x_basis_edge.float())
        distance, importance = _predicates(x)
        den = 1 + _strength(self.geo_den_delta)[None] * distance[..., None]
        den = den + _strength(self.geo_den_weight)[None] * importance[..., None]
        return basis, spine, edge_t / den, edge_x

    def geo_step(self, coords):
        _, spine, atten, edge_x = self.geometry_terms(coords)
        return (spine + (atten * edge_x).sum(2)).to(coords.dtype)

    def forward(self, x):
        coords = self.down(x).float().reshape(-1, self.num_subspaces, self.subspace_rank)
        for _ in range(self.geo_steps):
            coords = _coordinates(coords)
            coords = coords + self.geo_step(coords)
        coords = coords.reshape(*x.shape[:-1], -1).to(self.up.weight.dtype)
        return self.up(coords)

    @staticmethod
    def warp(x, raw_scale, raw_shift):
        return KS2Cell.warp(x.float(), raw_scale.float(), raw_shift.float())

    @torch.no_grad()
    def calibrate(self, batches, margin=1.2):
        """Freeze per-chart activation ranges from representative input batches."""
        if not math.isfinite(margin) or margin < 1:
            raise ValueError('margin must be finite and at least one')
        ranges = [torch.zeros_like(self.range_basis) for _ in range(3)]
        seen = False
        for x in batches:
            if not x.numel():
                continue
            seen = True
            coords = self.down(x).float().reshape(-1, self.num_subspaces, self.subspace_rank)
            for _ in range(self.geo_steps):
                coords = _coordinates(coords)
                basis, spine, atten, edge_x = self.geometry_terms(coords)
                for dest, value in zip(ranges, (basis, atten, edge_x)):
                    dest.copy_(torch.maximum(dest, value.transpose(0, 1).reshape(self.num_subspaces, -1).abs().amax(1)))
                coords = coords + spine + (atten * edge_x).sum(2)
        if not seen:
            raise ValueError('Calibration requires nonempty representative data')
        if not all(torch.isfinite(v).all() for v in ranges):
            raise ValueError('Calibration produced nonfinite ranges')
        for name, value in zip(('basis', 'atten', 'ex'), ranges):
            getattr(self, 'range_' + name).copy_(value.clamp_min(1e-8) * margin)
        return self
