"""A direct K-Splanifold 2 map and a cell with learned down/up projections."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class KS2(nn.Module):
    """Map (..., in_features) to (..., out_features) with one KS2 subspace.

    Input coordinates directly define the subspace. A single evaluation adds
    the learned spine and damped edges to produce output coordinates.

    Example:
        surface = KS2(in_features=4, out_features=4)
        output = surface(torch.rand(256, 4))
    """

    def __init__(self, in_features: int, out_features: int | None = None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = in_features if out_features is None else out_features
        k, n = self.in_features, self.out_features

        # Basis coefficients: spine, longitudinal edges, coordinate edges.
        self.geo_spine = nn.Parameter(torch.randn(6, n) * 0.05)
        self.geo_t_basis_edge = nn.Parameter(torch.randn(6, k, n) * 0.05)
        self.geo_x_basis_edge = nn.Parameter(torch.randn(4, k, n) * 0.05)

        # Edge damping.
        self.geo_den_delta = nn.Parameter(torch.full((k, n), -4.0))
        self.geo_den_weight = nn.Parameter(torch.full((k, n), -4.0))

        # Warps for the mean and individual coordinates.
        self.geo_raw_scale = nn.Parameter(torch.full((1,), 0.75))
        self.geo_raw_shift = nn.Parameter(torch.zeros(1))
        self.geo_edge_raw_scale = nn.Parameter(torch.full((1,), 0.75))
        self.geo_edge_raw_shift = nn.Parameter(torch.zeros(1))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # Normalize the subspace; use uniform weights for near-zero sums.
        x = coords.float() if coords.dtype == torch.float16 else coords
        rank = x.shape[-1]
        total = x.sum(dim=-1, keepdim=True)
        t = total / rank
        delta = x - t
        near_zero = total.abs() < 1e-6
        divisor = torch.where(near_zero, torch.ones_like(total), total)
        weights = x / divisor
        weights = torch.where(near_zero, torch.full_like(weights, 1.0 / rank), weights)

        # Evaluate the spine and the two factors of each edge.
        spine_basis = self.spine_basis(t)
        edge_basis = self.edge_basis(coords)
        spine = torch.einsum("...m,mo->...o", spine_basis, self.geo_spine)
        edge_t = torch.einsum("...m,mio->...io", spine_basis, self.geo_t_basis_edge)
        edge_x = torch.einsum("...ik,kio->...io", edge_basis, self.geo_x_basis_edge)

        # Damp the edges before reducing over input coordinates.
        damping = (
            1
            + F.softplus(self.geo_den_delta) * delta.abs()[..., None]
            + F.softplus(self.geo_den_weight) * weights.square()[..., None]
        )
        edges = (edge_t * edge_x / damping).sum(dim=-2)
        return (spine + edges).to(coords.dtype)

    def spine_basis(self, t: torch.Tensor) -> torch.Tensor:
        """Four Hermite terms followed by warped Chebyshev T2 and T3."""
        z = self.warp(t, self.geo_raw_scale, self.geo_raw_shift)
        t2, t3 = t * t, t * t * t
        return torch.cat(
            (
                2 * t3 - 3 * t2 + 1,
                -2 * t3 + 3 * t2,
                t3 - 2 * t2 + t,
                t3 - t2,
                2 * z * z - 1,
                4 * z * z * z - 3 * z,
            ),
            dim=-1,
        )

    def edge_basis(self, coords: torch.Tensor) -> torch.Tensor:
        """Warped Chebyshev T0 through T3 for each input coordinate."""
        z = self.warp(
            coords.unsqueeze(-1), self.geo_edge_raw_scale, self.geo_edge_raw_shift
        )
        return torch.cat(
            (torch.ones_like(z), z, 2 * z * z - 1, 4 * z * z * z - 3 * z),
            dim=-1,
        )

    @staticmethod
    def warp(
        x: torch.Tensor, raw_scale: torch.Tensor, raw_shift: torch.Tensor
    ) -> torch.Tensor:
        scale = F.softplus(raw_scale) + 1e-4
        shift = 0.5 * torch.tanh(raw_shift)
        return torch.tanh(scale * (x + shift))


class KS2Cell(nn.Module):
    """Map (..., in_features) to (..., out_features) through learned subspaces.

    The down projection produces ``num_subspaces`` groups of ``subspace_rank``
    coordinates. Each of ``geo_steps`` shared-parameter steps refines these
    groups, then the up projection maps them to the output feature space.
    ``out_features`` defaults to ``in_features``.

    Example:
        cell = KS2Cell(64, num_subspaces=4, subspace_rank=8)
        output = cell(torch.randn(16, 64))
    """

    def __init__(
        self,
        in_features: int,
        out_features: int | None = None,
        num_subspaces: int = 4,
        subspace_rank: int = 8,
        geo_steps: int = 2,
    ) -> None:
        super().__init__()
        self.num_subspaces = num_subspaces
        self.subspace_rank = subspace_rank
        self.geo_steps = geo_steps
        s, r = num_subspaces, subspace_rank

        # Input axes and output directions for all subspaces.
        out_features = in_features if out_features is None else out_features
        self.down = nn.Linear(in_features, s * r, bias=False)
        self.up = nn.Linear(s * r, out_features, bias=False)

        # Basis coefficients: spine, longitudinal edges, coordinate edges.
        self.geo_spine = nn.Parameter(torch.randn(s, 6, r) * 0.05)
        self.geo_t_basis_edge = nn.Parameter(torch.randn(s, 6, r, r) * 0.05)
        self.geo_x_basis_edge = nn.Parameter(torch.randn(s, 4, r, r) * 0.05)

        # Edge damping.
        self.geo_den_delta = nn.Parameter(torch.full((s, r, r), -4.0))
        self.geo_den_weight = nn.Parameter(torch.full((s, r, r), -4.0))

        # Per-subspace warps for the mean and individual coordinates.
        self.geo_raw_scale = nn.Parameter(torch.full((s,), 0.75))
        self.geo_raw_shift = nn.Parameter(torch.zeros(s))
        self.geo_edge_raw_scale = nn.Parameter(torch.full((s,), 0.75))
        self.geo_edge_raw_shift = nn.Parameter(torch.zeros(s))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coords = self.down(x).reshape(-1, self.num_subspaces, self.subspace_rank)
        for _ in range(self.geo_steps):
            coords = coords + self.geo_step(coords)
        coords = coords.reshape(*x.shape[:-1], self.num_subspaces * self.subspace_rank)
        return self.up(coords)

    def geo_step(self, coords: torch.Tensor) -> torch.Tensor:
        # Normalize each subspace; use uniform weights for near-zero sums.
        x = coords.float() if coords.dtype == torch.float16 else coords
        rank = x.shape[-1]
        total = x.sum(dim=-1, keepdim=True)
        t = total / rank
        delta = x - t
        near_zero = total.abs() < 1e-6
        divisor = torch.where(near_zero, torch.ones_like(total), total)
        weights = x / divisor
        weights = torch.where(near_zero, torch.full_like(weights, 1.0 / rank), weights)

        # Evaluate the spine and the two factors of each edge.
        spine_basis = self.spine_basis(t)
        edge_basis = self.edge_basis(coords)

        spine = torch.einsum("nsm,smr->nsr", spine_basis, self.geo_spine)
        edge_t = torch.einsum("nsm,smro->nsro", spine_basis, self.geo_t_basis_edge)
        edge_x = torch.einsum("nsrk,skro->nsro", edge_basis, self.geo_x_basis_edge)

        # Damp the edges before reducing over input coordinates.
        damping = (
            1
            + F.softplus(self.geo_den_delta)[None] * delta.abs()[..., None]
            + F.softplus(self.geo_den_weight)[None] * weights.square()[..., None]
        )
        edges = (edge_t * edge_x / damping).sum(dim=2)
        return (spine + edges).to(coords.dtype)

    def spine_basis(self, t: torch.Tensor) -> torch.Tensor:
        """Four Hermite terms followed by warped Chebyshev T2 and T3."""
        z = self.warp(t, self.geo_raw_scale, self.geo_raw_shift)
        t2, t3 = t * t, t * t * t
        return torch.cat(
            (
                2 * t3 - 3 * t2 + 1,
                -2 * t3 + 3 * t2,
                t3 - 2 * t2 + t,
                t3 - t2,
                2 * z * z - 1,
                4 * z * z * z - 3 * z,
            ),
            dim=-1,
        )

    def edge_basis(self, coords: torch.Tensor) -> torch.Tensor:
        """Warped Chebyshev T0 through T3 for each input coordinate."""
        z = self.warp(
            coords.unsqueeze(-1), self.geo_edge_raw_scale, self.geo_edge_raw_shift
        )
        return torch.cat(
            (torch.ones_like(z), z, 2 * z * z - 1, 4 * z * z * z - 3 * z),
            dim=-1,
        )

    @staticmethod
    def warp(
        x: torch.Tensor, raw_scale: torch.Tensor, raw_shift: torch.Tensor
    ) -> torch.Tensor:
        shape = (1, -1) + (1,) * (x.ndim - 2)
        scale = F.softplus(raw_scale).view(shape) + 1e-4
        shift = 0.5 * torch.tanh(raw_shift).view(shape)
        return torch.tanh(scale * (x + shift))

