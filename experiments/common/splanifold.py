from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch


def _hermite(
    p0: torch.Tensor,
    p1: torch.Tensor,
    v0: torch.Tensor,
    v1: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1


def splanifold_map(
    r: torch.Tensor,
    anchor_start: torch.Tensor,
    anchor_end: torch.Tensor,
    basis_start: torch.Tensor,
    basis_end: torch.Tensor,
    pos_tangent_start: torch.Tensor,
    pos_tangent_end: torch.Tensor,
    basis_tangent_start: torch.Tensor,
    basis_tangent_end: torch.Tensor,
    sigma: torch.Tensor | float,
    extrapolation: torch.Tensor | float,
) -> torch.Tensor:
    """
    Reference implementation from the prompt.

    Shapes:
      r: (..., K)
      anchors: (N,)
      basis_*: (K, N)
      pos_tangent_*: (K, N)
      basis_tangent_*: (K, N)
    """
    k = r.shape[-1]
    u = r * (1.0 + 2.0 * extrapolation) - extrapolation
    sigma_sum = u.sum(dim=-1, keepdim=True)
    t = sigma_sum / float(k)
    delta = u - t

    sigma_safe = torch.where(sigma_sum == 0, torch.ones_like(sigma_sum), sigma_sum)
    w = u / sigma_safe
    w = torch.where(sigma_sum == 0, torch.full_like(w, 1.0 / float(k)), w)

    v_start = sigma * (w.unsqueeze(-1) * (pos_tangent_start - anchor_start.unsqueeze(-2))).sum(dim=-2)
    v_end = sigma * (w.unsqueeze(-1) * (pos_tangent_end - anchor_end.unsqueeze(-2))).sum(dim=-2)

    d_start = (delta.unsqueeze(-1) * basis_start).sum(dim=-2)
    d_end = (delta.unsqueeze(-1) * basis_end).sum(dim=-2)

    t_start = sigma * (delta.unsqueeze(-1) * (basis_tangent_start - basis_start)).sum(dim=-2)
    t_end = sigma * (delta.unsqueeze(-1) * (basis_tangent_end - basis_end)).sum(dim=-2)

    spine = _hermite(anchor_start, anchor_end, v_start, v_end, t)
    disp = _hermite(d_start, d_end, t_start, t_end, t)
    return spine + disp


def weights_regularized(u: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Eq. (7) weight regularization from the paper."""
    sigma = u.sum(dim=-1, keepdim=True)
    k = u.shape[-1]
    eps2 = epsilon * epsilon
    return (sigma * u + eps2 / float(k)) / (sigma * sigma + eps2)


def warp_delta(delta: torch.Tensor, delta_max: float) -> torch.Tensor:
    if math.isinf(delta_max):
        return delta
    norm = torch.linalg.norm(delta, dim=-1, keepdim=True)
    return delta / torch.sqrt(1.0 + (norm / delta_max) ** 2)


def unwarp_delta(delta_tilde: torch.Tensor, delta_max: float) -> torch.Tensor:
    if math.isinf(delta_max):
        return delta_tilde
    norm = torch.linalg.norm(delta_tilde, dim=-1, keepdim=True)
    return delta_tilde / torch.sqrt(1.0 - (norm / delta_max) ** 2)


@dataclass
class K2ExampleControls:
    anchor_start: torch.Tensor
    anchor_end: torch.Tensor
    basis_start: torch.Tensor
    basis_end: torch.Tensor
    pos_tangent_start: torch.Tensor
    pos_tangent_end: torch.Tensor
    basis_tangent_start: torch.Tensor
    basis_tangent_end: torch.Tensor
    sigma: float
    extrapolation: float


def make_k2_example_controls(dtype: torch.dtype = torch.float32) -> K2ExampleControls:
    """
    Replicates the initial K=2 control setup in k-splanifolds-2D-to-3D-toy.html.
    """
    width_scale = 0.5
    anchor_start = torch.tensor([0.0, -2.0, 0.0], dtype=dtype)
    anchor_end = torch.tensor([0.0, 2.0, 0.0], dtype=dtype)

    basis_start = torch.tensor(
        [[width_scale, 0.0, 0.0], [0.0, width_scale, 0.0]], dtype=dtype
    )
    basis_end = basis_start.clone()

    offsets_start = torch.tensor([[2.0, 2.0, 0.0], [-2.0, 2.0, 0.0]], dtype=dtype)
    offsets_end = torch.tensor([[2.0, -2.0, 0.0], [-2.0, -2.0, 0.0]], dtype=dtype)
    pos_tangent_start = anchor_start + offsets_start
    pos_tangent_end = anchor_end + offsets_end

    theta = 0.4
    basis_tangent_start = torch.tensor(
        [
            [math.cos(theta) * width_scale, math.sin(theta) * width_scale, 0.0],
            [math.sin(theta) * width_scale, math.cos(theta) * width_scale, 0.0],
        ],
        dtype=dtype,
    )
    basis_tangent_end = basis_end.clone()

    sigma = 3.0
    extrapolation = 0.0

    return K2ExampleControls(
        anchor_start=anchor_start,
        anchor_end=anchor_end,
        basis_start=basis_start,
        basis_end=basis_end,
        pos_tangent_start=pos_tangent_start,
        pos_tangent_end=pos_tangent_end,
        basis_tangent_start=basis_tangent_start,
        basis_tangent_end=basis_tangent_end,
        sigma=sigma,
        extrapolation=extrapolation,
    )


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
