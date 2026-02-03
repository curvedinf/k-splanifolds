from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from experiments.common.splanifold import seed_all, splanifold_map, weights_regularized


def _hermite(p0, p1, v0, v1, t):
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1


def make_cubic_coeffs(seed: int = 0) -> Dict[str, torch.Tensor]:
    rng = torch.Generator().manual_seed(seed)
    a = [torch.randn(4, generator=rng) * 0.5 for _ in range(4)]
    B0 = torch.randn(4, 4, generator=rng) * 0.5
    B1 = torch.randn(4, 4, generator=rng) * 0.5
    B2 = torch.randn(4, 4, generator=rng) * 0.5
    return {"a": a, "B0": B0, "B1": B1, "B2": B2}


def cubic_target(r: torch.Tensor, coeffs: Dict[str, torch.Tensor]) -> torch.Tensor:
    t = r.mean(dim=-1, keepdim=True)
    delta = r - t

    a0, a1, a2, a3 = coeffs["a"]
    poly = a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3

    B0 = coeffs["B0"]
    B1 = coeffs["B1"]
    B2 = coeffs["B2"]

    B = B0 + B1 * t.unsqueeze(-1) + B2 * (t ** 2).unsqueeze(-1)
    linear = torch.bmm(B.expand(r.shape[0], -1, -1), delta.unsqueeze(-1)).squeeze(-1)
    return poly + linear


def make_fourier_params(seed: int = 0, m: int = 64) -> Dict[str, torch.Tensor]:
    rng = torch.Generator().manual_seed(seed)
    omega = torch.rand(m, generator=rng) * 3.0 + 1.0
    k = torch.randn(m, 4, generator=rng)
    k = k / (torch.linalg.norm(k, dim=-1, keepdim=True) + 1e-8)
    phi = torch.rand(m, generator=rng) * 2.0 * math.pi
    v = torch.randn(m, 4, generator=rng) * 0.5
    return {"omega": omega, "k": k, "phi": phi, "v": v}


def irregular_target(r: torch.Tensor, base: torch.Tensor, params: Dict[str, torch.Tensor], gamma: float = 0.15) -> torch.Tensor:
    phase = 2.0 * math.pi * (r @ params["k"].T) * params["omega"].unsqueeze(0) + params["phi"].unsqueeze(0)
    mix = torch.sin(phase) @ params["v"]
    return base + gamma * mix


def build_features(r: torch.Tensor, epsilon: float) -> torch.Tensor:
    k = r.shape[-1]
    u = r
    sigma = u.sum(dim=-1, keepdim=True)
    t = sigma / float(k)
    delta = u - t
    w = weights_regularized(u, epsilon=epsilon)

    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h01 = -2.0 * t3 + 3.0 * t2
    h10 = t3 - 2.0 * t2 + t
    h11 = t3 - t2

    f0 = h00
    f1 = h01
    f2 = h10 * w
    f3 = h11 * w
    f4 = h00 * delta
    f5 = h01 * delta
    f6 = h10 * delta
    f7 = h11 * delta

    return torch.cat([f0, f1, f2, f3, f4, f5, f6, f7], dim=-1)


def fit_splanifold(r: torch.Tensor, x: torch.Tensor, epsilon: float = 1e-4, ridge: float = 1e-4) -> torch.Tensor:
    # Solve min ||C F - X||_F with optional ridge via augmentation for numerical stability.
    F = build_features(r, epsilon=epsilon).T.double()
    X = x.T.double()

    if ridge > 0:
        dim = F.shape[0]
        eye = torch.eye(dim, dtype=F.dtype)
        F_aug = torch.cat([F, math.sqrt(ridge) * eye], dim=1)
        X_aug = torch.cat([X, torch.zeros(X.shape[0], dim, dtype=X.dtype)], dim=1)
        C = torch.linalg.lstsq(F_aug.T, X_aug.T).solution.T
    else:
        C = torch.linalg.lstsq(F.T, X.T).solution.T

    return C.to(x.dtype)


def eval_splanifold(r: torch.Tensor, C: torch.Tensor, k: int = 4, n: int = 4) -> torch.Tensor:
    idx = 0
    P0 = C[:, idx]
    idx += 1
    P1 = C[:, idx]
    idx += 1
    P0p = C[:, idx : idx + k]
    idx += k
    P1p = C[:, idx : idx + k]
    idx += k
    E0 = C[:, idx : idx + k]
    idx += k
    E1 = C[:, idx : idx + k]
    idx += k
    E0p = C[:, idx : idx + k]
    idx += k
    E1p = C[:, idx : idx + k]

    anchor_start = P0
    anchor_end = P1
    basis_start = E0.T
    basis_end = E1.T
    pos_tangent_start = anchor_start + P0p.T
    pos_tangent_end = anchor_end + P1p.T
    basis_tangent_start = basis_start + E0p.T
    basis_tangent_end = basis_end + E1p.T

    return splanifold_map(
        r,
        anchor_start,
        anchor_end,
        basis_start,
        basis_end,
        pos_tangent_start,
        pos_tangent_end,
        basis_tangent_start,
        basis_tangent_end,
        sigma=1.0,
        extrapolation=0.0,
    )


def quantize_tensor(x: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    qmax = 2 ** (bits - 1) - 1
    if qmax <= 0:
        return torch.zeros_like(x), torch.zeros_like(x, dtype=torch.int32)
    scale = x.abs().max()
    if scale == 0:
        return torch.zeros_like(x), torch.zeros_like(x, dtype=torch.int32)
    q = torch.clamp(torch.round(x / scale * qmax), -qmax, qmax).to(torch.int32)
    xq = q.to(x.dtype) / float(qmax) * scale
    return xq, q


def entropy_bits(q: torch.Tensor) -> float:
    if q.numel() == 0:
        return 0.0
    values, counts = torch.unique(q, return_counts=True)
    p = counts.float() / counts.sum()
    return float(-(p * torch.log2(p)).sum())


def quantize_blocks(blocks: List[torch.Tensor], bits: int) -> Tuple[List[torch.Tensor], float]:
    total = sum(b.numel() for b in blocks)
    accum = 0.0
    q_blocks = []
    for block in blocks:
        q_block, q_codes = quantize_tensor(block, bits)
        H = entropy_bits(q_codes)
        accum += block.numel() * H
        q_blocks.append(q_block)
    return q_blocks, accum / float(total)


class MLP(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def train_mlp(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    steps: int,
    lr: float,
    weight_decay: float = 1e-4,
) -> Tuple[MLP, float]:
    model = MLP(hidden=64)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = model(x_train)
        loss = torch.mean((pred - y_train) ** 2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        test_rmse = rmse(model(x_test), y_test)
    return model, test_rmse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    steps = 200 if args.fast else args.steps

    out_dir = Path(__file__).parent

    seed_all(1234)
    coeffs = make_cubic_coeffs(0)
    fourier_params = make_fourier_params(0)

    def run_task(task_name: str, noise_std: float, irregular: bool):
        r_train = torch.rand(256, 4)
        r_test = torch.rand(256, 4)

        base_train = cubic_target(r_train, coeffs)
        base_test = cubic_target(r_test, coeffs)

        if irregular:
            y_train_clean = irregular_target(r_train, base_train, fourier_params)
            y_test_clean = irregular_target(r_test, base_test, fourier_params)
        else:
            y_train_clean = base_train
            y_test_clean = base_test

        y_train = y_train_clean + torch.randn_like(y_train_clean) * noise_std

        # Fit K-Splanifold (two-node, K=N=4)
        C = fit_splanifold(r_train, y_train)
        pred = eval_splanifold(r_test, C)
        sp_float_rmse = rmse(pred, y_test_clean)

        # Fit MLP
        mlp, mlp_float_rmse = train_mlp(r_train, y_train, r_test, y_test_clean, steps=steps, lr=args.lr)

        return {
            "r_train": r_train,
            "r_test": r_test,
            "y_test_clean": y_test_clean,
            "C": C,
            "sp_float_rmse": sp_float_rmse,
            "mlp": mlp,
            "mlp_float_rmse": mlp_float_rmse,
        }

    tasks = {
        "cubic": run_task("cubic", noise_std=0.07, irregular=False),
        "irregular": run_task("irregular", noise_std=0.18, irregular=True),
    }

    rows = []
    curve_rows = []
    for task_name, data in tasks.items():
        r_test = data["r_test"]
        y_test_clean = data["y_test_clean"]

        # Cache float MLP weights for quantization sweeps
        mlp = data["mlp"]
        mlp_state = [p.detach().clone() for p in mlp.parameters()]

        # Splanifold quantization sweep
        C = data["C"]
        blocks = []
        k = 4
        idx = 0
        P0 = C[:, idx]
        idx += 1
        P1 = C[:, idx]
        idx += 1
        P0p = C[:, idx : idx + k]
        idx += k
        P1p = C[:, idx : idx + k]
        idx += k
        E0 = C[:, idx : idx + k]
        idx += k
        E1 = C[:, idx : idx + k]
        idx += k
        E0p = C[:, idx : idx + k]
        idx += k
        E1p = C[:, idx : idx + k]
        blocks = [P0, P1, P0p, P1p, E0, E1, E0p, E1p]

        sp_curve = []
        mlp_curve = []

        for bits in range(2, 11):
            q_blocks, hpp = quantize_blocks(blocks, bits)
            qP0, qP1, qP0p, qP1p, qE0, qE1, qE0p, qE1p = q_blocks
            Cq = torch.cat(
                [
                    qP0.unsqueeze(1),
                    qP1.unsqueeze(1),
                    qP0p,
                    qP1p,
                    qE0,
                    qE1,
                    qE0p,
                    qE1p,
                ],
                dim=1,
            )
            pred = eval_splanifold(r_test, Cq)
            sp_rmse = rmse(pred, y_test_clean)
            sp_curve.append((hpp, sp_rmse, bits))

            # MLP quantization
            mlp_blocks = [p.detach().clone() for p in mlp_state]
            q_mlp_blocks, hpp_mlp = quantize_blocks(mlp_blocks, bits)
            with torch.no_grad():
                for p, q in zip(mlp.parameters(), q_mlp_blocks):
                    p.copy_(q)
                mlp_rmse = rmse(mlp(r_test), y_test_clean)
            mlp_curve.append((hpp_mlp, mlp_rmse, bits))

            # Restore float weights
            with torch.no_grad():
                for p, orig in zip(mlp.parameters(), mlp_state):
                    p.copy_(orig)

        sp_curve = sorted(sp_curve, key=lambda x: x[0])
        mlp_curve = sorted(mlp_curve, key=lambda x: x[0])

        for hpp, rmse_val, bits in sp_curve:
            curve_rows.append(
                {
                    "Task": task_name,
                    "Model": "K-Splanifold (K=N=4)",
                    "bits": bits,
                    "Hpp": f"{hpp:.4f}",
                    "RMSE": f"{rmse_val:.4f}",
                }
            )
        for hpp, rmse_val, bits in mlp_curve:
            curve_rows.append(
                {
                    "Task": task_name,
                    "Model": "MLP (1x64)",
                    "bits": bits,
                    "Hpp": f"{hpp:.4f}",
                    "RMSE": f"{rmse_val:.4f}",
                }
            )

        # Find b* (within 5% of float RMSE)
        sp_float = data["sp_float_rmse"]
        mlp_float = data["mlp_float_rmse"]

        sp_best = next((c for c in sp_curve if c[1] <= 1.05 * sp_float), sp_curve[-1])
        mlp_best = next((c for c in mlp_curve if c[1] <= 1.05 * mlp_float), mlp_curve[-1])

        rows.append(
            {
                "Task": task_name,
                "Model": "K-Splanifold (K=N=4)",
                "P": 104,
                "RMSE (float)": f"{sp_float:.4f}",
                "b*": sp_best[2],
                "RMSE (b*)": f"{sp_best[1]:.4f}",
                "Hpp (b*)": f"{sp_best[0]:.2f}",
                "Size (bytes)": f"{104 * sp_best[0] / 8.0:.1f}",
            }
        )
        rows.append(
            {
                "Task": task_name,
                "Model": "MLP (1x64)",
                "P": sum(p.numel() for p in data["mlp"].parameters()),
                "RMSE (float)": f"{mlp_float:.4f}",
                "b*": mlp_best[2],
                "RMSE (b*)": f"{mlp_best[1]:.4f}",
                "Hpp (b*)": f"{mlp_best[0]:.2f}",
                "Size (bytes)": f"{sum(p.numel() for p in data['mlp'].parameters()) * mlp_best[0] / 8.0:.1f}",
            }
        )

    table_path = out_dir / "table4_entropy.csv"
    with table_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Task", "Model", "P", "RMSE (float)", "b*", "RMSE (b*)", "Hpp (b*)", "Size (bytes)"],
        )
        writer.writeheader()
        writer.writerows(rows)

    curve_path = out_dir / "figure10_entropy.csv"
    with curve_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Task", "Model", "bits", "Hpp", "RMSE"],
        )
        writer.writeheader()
        writer.writerows(curve_rows)

    # Plotting removed; CSV outputs only.


if __name__ == "__main__":
    main()
