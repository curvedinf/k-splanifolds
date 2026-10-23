"""KS2 block cell — reimplemented for RHAC.

KS2 is the geo-cell algorithm selected by a large candidate search
(hundreds of candidates) as the best loss across a variety of predictions;
reimplemented here so the fused kernel can be customized.

What is preserved exactly:
  - ``_safe_weight_map`` coordinate decomposition (weights w, spine t, deviation delta)
  - hybrid spine basis: 4 cubic-Hermite terms + 2 Chebyshev terms of the
    tanh-warped spine (learned per-cell scale/shift)
  - 4-term Chebyshev edge basis over tanh-warped raw coordinates
  - bilinear edge numerator ``t_proj * x_proj`` over the rational denominator
    ``1 + softplus(den_delta)|delta| + softplus(den_weight) w^2``
  - ``geo_steps`` residual iteration ``x <- x + eval(x)``

RHAC adaptations (math of the cell unchanged):
  - operates on the FULL K-dim activation block as one rank-K coordinate
    (no subspace decomposition; equivalent to num_subspaces=1)
  - parameters batched over the G activation blocks of a layer as leading
    tensor dims — one launch per layer, kernel-shaped from the start
  - bf16 parameter storage, fp32 math via upcast in forward
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def ks2_local_param_count(blocks: int, K: int, include_out_scale: bool = True) -> int:
    """Per-layer parameter count of the batched KS2 bank.

    spine 6K + t-edge 6K^2 + x-edge 4K^2 + den_delta K^2 + den_weight K^2
    + 4 scalars, all times G blocks (+ G for the zero-init output scale).
    """
    per_block = 12 * K * K + 6 * K + 4
    return blocks * (per_block + (1 if include_out_scale else 0))


def _safe_weight_map(x: torch.Tensor):
    rank = x.shape[-1]
    sum_x = x.sum(dim=-1, keepdim=True)
    t = sum_x / float(rank)
    delta = x - t
    small = sum_x.abs() < 1e-6
    safe_sum = torch.where(small, torch.ones_like(sum_x), sum_x)
    w = x / safe_sum
    w = torch.where(small.expand_as(w), torch.full_like(w, 1.0 / float(rank)), w)
    return w, t, delta


def _warped_coordinate(x: torch.Tensor, raw_scale: torch.Tensor, raw_shift: torch.Tensor, g_dim: int) -> torch.Tensor:
    """tanh-warp with per-block scalars broadcast along the G axis (`g_dim`)."""
    shape = [1] * x.dim()
    shape[g_dim] = -1
    scale = F.softplus(raw_scale).view(shape) + 1e-4
    shift = 0.5 * torch.tanh(raw_shift).view(shape)
    return torch.tanh(scale * (x + shift))


def _hermite_basis(t: torch.Tensor) -> torch.Tensor:
    t2 = t * t
    t3 = t2 * t
    return torch.cat(
        [
            2.0 * t3 - 3.0 * t2 + 1.0,
            -2.0 * t3 + 3.0 * t2,
            t3 - 2.0 * t2 + t,
            t3 - t2,
        ],
        dim=-1,
    )


def _chebyshev_basis(z: torch.Tensor, basis_size: int) -> torch.Tensor:
    if basis_size <= 0:
        raise ValueError("basis_size must be positive.")
    if basis_size == 1:
        return torch.ones_like(z)
    basis = [torch.ones_like(z), z]
    while len(basis) < basis_size:
        basis.append(2.0 * z * basis[-1] - basis[-2])
    return torch.cat(basis[:basis_size], dim=-1)


class KS2BlockBank(nn.Module):
    """Batched bank of G rank-K KS2 cells (one cell per activation block).

    Parameters (leading dim = G blocks):
      geo_spine         (G, 6, K)
      geo_t_basis_edge  (G, 6, K, K)
      geo_x_basis_edge  (G, 4, K, K)
      geo_den_delta     (G, K, K)
      geo_den_weight    (G, K, K)
      geo_raw_scale/shift, geo_edge_raw_scale/shift  (G,)
      out_scale         (G,)  — zero-initialized; anchors function preservation

    forward: coords (..., G, K) [fp32] -> same shape, the geo_steps-iterated
    residual map x + eval(x).
    """

    def __init__(self, blocks: int, K: int, geo_steps: int = 2, dtype: torch.dtype = torch.bfloat16,
                 seed: int = 0):
        super().__init__()
        self.blocks, self.K, self.geo_steps = blocks, K, max(1, geo_steps)
        gen = torch.Generator().manual_seed(seed)
        scale = 0.05

        def _randn(*shape):
            return (torch.randn(*shape, generator=gen) * scale).to(dtype)

        self.geo_spine = nn.Parameter(_randn(blocks, 6, K))
        self.geo_t_basis_edge = nn.Parameter(_randn(blocks, 6, K, K))
        self.geo_x_basis_edge = nn.Parameter(_randn(blocks, 4, K, K))
        self.geo_den_delta = nn.Parameter(torch.full((blocks, K, K), -4.0, dtype=dtype))
        self.geo_den_weight = nn.Parameter(torch.full((blocks, K, K), -4.0, dtype=dtype))
        self.geo_raw_scale = nn.Parameter(torch.full((blocks,), 0.75, dtype=dtype))
        self.geo_raw_shift = nn.Parameter(torch.zeros(blocks, dtype=dtype))
        self.geo_edge_raw_scale = nn.Parameter(torch.full((blocks,), 0.75, dtype=dtype))
        self.geo_edge_raw_shift = nn.Parameter(torch.zeros(blocks, dtype=dtype))
        self.out_scale = nn.Parameter(torch.zeros(blocks, dtype=dtype))

    def _eval_once(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (..., G, K) fp32; parameters upcast to fp32 for the math.
        w, t, delta = _safe_weight_map(coords)

        z = _warped_coordinate(t, self.geo_raw_scale.float(), self.geo_raw_shift.float(), g_dim=-2)
        cheby_all = _chebyshev_basis(z, 4)
        extra_basis = torch.cat([cheby_all[..., 2:3], cheby_all[..., 3:4]], dim=-1)
        t_basis = torch.cat([_hermite_basis(t), extra_basis], dim=-1)  # (...,G,6)

        edge_coord = _warped_coordinate(
            coords.unsqueeze(-1), self.geo_edge_raw_scale.float(), self.geo_edge_raw_shift.float(), g_dim=-3
        )
        edge_basis = _chebyshev_basis(edge_coord, 4)  # (...,G,K,4)

        spine = torch.einsum("...gm,gmr->...gr", t_basis, self.geo_spine.float())
        t_proj = torch.einsum("...gm,gmro->...gro", t_basis, self.geo_t_basis_edge.float())
        x_proj = torch.einsum("...grk,gkro->...gro", edge_basis, self.geo_x_basis_edge.float())
        numerator = t_proj * x_proj

        denom = 1.0
        denom = denom + F.softplus(self.geo_den_delta.float()).unsqueeze(0) * delta.abs().unsqueeze(-1)
        denom = denom + F.softplus(self.geo_den_weight.float()).unsqueeze(0) * w.square().unsqueeze(-1)
        return spine + (numerator / denom).sum(dim=-2)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords.float()
        for _ in range(self.geo_steps):
            x = x + self._eval_once(x)
        return x
