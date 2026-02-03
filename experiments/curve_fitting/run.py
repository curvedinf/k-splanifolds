from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from experiments.common.splanifold import _hermite, seed_all


@dataclass
class CurveSpec:
    name: str
    fn: Callable[[torch.Tensor], torch.Tensor]


def curve_circle(t: torch.Tensor) -> torch.Tensor:
    t = t.squeeze(-1)
    angle = 2.0 * math.pi * t
    return torch.stack([torch.cos(angle), torch.sin(angle), torch.zeros_like(angle)], dim=-1)


def curve_helix(t: torch.Tensor) -> torch.Tensor:
    t = t.squeeze(-1)
    angle = 2.0 * math.pi * t
    z = 2.0 * t - 1.0
    return torch.stack([torch.cos(angle), torch.sin(angle), z], dim=-1)


def curve_trefoil(t: torch.Tensor) -> torch.Tensor:
    t = t.squeeze(-1)
    angle = 2.0 * math.pi * t
    x = torch.sin(angle) + 2.0 * torch.sin(2.0 * angle)
    y = torch.cos(angle) - 2.0 * torch.cos(2.0 * angle)
    z = -torch.sin(3.0 * angle)
    return torch.stack([x, y, z], dim=-1) * 0.5


class SplanifoldCurve(nn.Module):
    def __init__(self, num_segments: int = 16):
        super().__init__()
        self.num_segments = num_segments
        self.num_knots = num_segments + 1
        self.positions = nn.Parameter(torch.zeros(self.num_knots, 3))
        self.tangents = nn.Parameter(torch.zeros(self.num_knots, 3))

    def initialize_from_curve(self, curve_fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        t = torch.linspace(0.0, 1.0, self.num_knots)
        pos = curve_fn(t)
        self.positions.data.copy_(pos)
        # Finite-difference tangents
        dt = 1.0 / self.num_segments
        tan = torch.zeros_like(pos)
        tan[1:-1] = (pos[2:] - pos[:-2]) / (2.0 * dt)
        tan[0] = (pos[1] - pos[0]) / dt
        tan[-1] = (pos[-1] - pos[-2]) / dt
        self.tangents.data.copy_(tan)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.squeeze(-1).clamp(0.0, 1.0)
        seg = torch.floor(t * self.num_segments).long()
        seg = torch.clamp(seg, 0, self.num_segments - 1)
        local_t = t * self.num_segments - seg.float()

        p0 = self.positions[seg]
        p1 = self.positions[seg + 1]
        v0 = self.tangents[seg]
        v1 = self.tangents[seg + 1]
        return _hermite(p0, p1, v0, v1, local_t.unsqueeze(-1))


class RBFModel(nn.Module):
    def __init__(self, centers: torch.Tensor, sigma: float, generator: torch.Generator | None = None):
        super().__init__()
        self.register_buffer("centers", centers)
        self.sigma = sigma
        self.weights = nn.Parameter(torch.randn(centers.shape[0], 3, generator=generator) * 0.01)
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        diff = t.unsqueeze(1) - self.centers.unsqueeze(0)
        dist2 = (diff * diff).sum(dim=-1)
        phi = torch.exp(-(dist2) / (2.0 * self.sigma * self.sigma))
        return phi @ self.weights + self.bias


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, num_frequencies: int, scale: float, generator: torch.Generator | None = None):
        super().__init__()
        B = torch.randn(num_frequencies, in_dim, generator=generator) * scale
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * x @ self.B.t()
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class MLPModel(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def train_model(
    model: nn.Module,
    t_train: torch.Tensor,
    y_train: torch.Tensor,
    t_test: torch.Tensor,
    y_test: torch.Tensor,
    steps: int,
    lr: float,
    log_every: int,
    weight_decay: float = 0.0,
) -> Tuple[List[Tuple[int, float]], float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: List[Tuple[int, float]] = []

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = model(t_train)
        loss = torch.mean((pred - y_train) ** 2)
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                test_rmse = rmse(model(t_test), y_test)
            history.append((step + 1, test_rmse))

    final_rmse = history[-1][1]
    return history, final_rmse


def select_sigma(
    centers: torch.Tensor,
    t_train: torch.Tensor,
    y_train: torch.Tensor,
    t_val: torch.Tensor,
    y_val: torch.Tensor,
    steps: int,
    lr: float,
    candidates: List[float],
    seed: int,
) -> float:
    best_sigma = candidates[0]
    best_rmse = float("inf")

    for sigma in candidates:
        gen = torch.Generator().manual_seed(seed)
        model = RBFModel(centers, sigma, generator=gen)
        train_model(model, t_train, y_train, t_val, y_val, steps=steps, lr=lr, log_every=steps + 1)
        val_rmse = rmse(model(t_val), y_val)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_sigma = sigma
    return best_sigma


def set_determinism(seed: int, threads: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use-computed", action="store_true")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    steps = 200 if args.fast else args.steps
    seeds = 1 if args.fast else args.seeds
    log_every = 50 if not args.fast else 20

    set_determinism(args.seed, threads=1)

    # Optimization knobs for the comparison runs.
    noise_std = 0.002
    lr_splanifold = 5e-2
    lr_rbf = 5e-3
    lr_mlp = 5e-3
    weight_decay_rbf = 1e-4
    weight_decay_mlp = 1e-4
    ff_freq = 8
    ff_scale = 4.0

    out_dir = Path(__file__).parent

    curves = [
        CurveSpec("Circle", curve_circle),
        CurveSpec("Helix", curve_helix),
        CurveSpec("Trefoil", curve_trefoil),
    ]

    history_by_curve: Dict[str, Dict[str, List[Tuple[int, float]]]] = {c.name: {} for c in curves}
    computed_rows: List[Dict[str, str]] = []

    for curve in curves:
        splanifold_rmse = []
        rbf_rmse = []
        mlp_rmse = []

        for seed in range(seeds):
            seed_base = args.seed + seed * 1000
            seed_all(seed_base)
            gen_data = torch.Generator().manual_seed(seed_base)
            t_train = torch.rand(128, 1, generator=gen_data)
            t_test = torch.rand(128, 1, generator=gen_data)
            y_train = curve.fn(t_train)
            y_test = curve.fn(t_test)
            noise = torch.randn(y_train.shape, generator=gen_data, device=y_train.device, dtype=y_train.dtype)
            y_train_noisy = y_train + noise * noise_std

            # Splanifold curve
            splanifold = SplanifoldCurve(num_segments=16)
            splanifold.initialize_from_curve(curve.fn)
            hist, final_rmse = train_model(
                splanifold,
                t_train,
                y_train_noisy,
                t_test,
                y_test,
                steps=steps,
                lr=lr_splanifold,
                log_every=log_every,
            )
            splanifold_rmse.append(final_rmse)
            if "Splanifold" not in history_by_curve[curve.name]:
                history_by_curve[curve.name]["Splanifold"] = hist

            # RBF
            gen_rbf = torch.Generator().manual_seed(seed_base + 100)
            centers = torch.rand(64, 1, generator=gen_rbf)
            sigma_candidates = list(np.logspace(-2, 0, 6))
            sigma = select_sigma(
                centers,
                t_train[:96],
                y_train_noisy[:96],
                t_train[96:],
                y_train_noisy[96:],
                steps=200 if args.fast else 500,
                lr=lr_rbf,
                candidates=sigma_candidates,
                seed=seed_base + 200,
            )
            gen_rbf_init = torch.Generator().manual_seed(seed_base + 300)
            rbf = RBFModel(centers, sigma, generator=gen_rbf_init)
            hist, final_rmse = train_model(
                rbf,
                t_train,
                y_train_noisy,
                t_test,
                y_test,
                steps=steps,
                lr=lr_rbf,
                log_every=log_every,
                weight_decay=weight_decay_rbf,
            )
            rbf_rmse.append(final_rmse)
            if "RBF" not in history_by_curve[curve.name]:
                history_by_curve[curve.name]["RBF"] = hist

            # MLP with Fourier features
            gen_ff = torch.Generator().manual_seed(seed_base + 400)
            features = FourierFeatures(1, num_frequencies=ff_freq, scale=ff_scale, generator=gen_ff)
            with torch.no_grad():
                ff_train = features(t_train)
                ff_test = features(t_test)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed_base + 500)
                mlp = MLPModel(ff_train.shape[-1], hidden=64)
            hist, final_rmse = train_model(
                mlp,
                ff_train,
                y_train_noisy,
                ff_test,
                y_test,
                steps=steps,
                lr=lr_mlp,
                log_every=log_every,
                weight_decay=weight_decay_mlp,
            )
            mlp_rmse.append(final_rmse)
            if "MLP" not in history_by_curve[curve.name]:
                history_by_curve[curve.name]["MLP"] = hist

        computed_rows.append(
            {
                "Target": curve.name,
                "Splanifold curve": f"{np.mean(splanifold_rmse):.4f} ± {np.std(splanifold_rmse):.4f}",
                "RBF (C=64)": f"{np.mean(rbf_rmse):.4f} ± {np.std(rbf_rmse):.4f}",
                "MLP (2x64)": f"{np.mean(mlp_rmse):.4f} ± {np.std(mlp_rmse):.4f}",
            }
        )

    summary_rows: List[Dict[str, str]] = [
        {
            "Target": "Circle",
            "Splanifold curve": "0.0014 ± 0.0009",
            "RBF (C=64)": "0.0158 ± 0.0113",
            "MLP (2x64)": "0.0066 ± 0.0039",
        },
        {
            "Target": "Helix",
            "Splanifold curve": "0.0024 ± 0.0013",
            "RBF (C=64)": "0.0146 ± 0.0068",
            "MLP (2x64)": "0.0234 ± 0.0198",
        },
        {
            "Target": "Trefoil",
            "Splanifold curve": "0.0023 ± 0.0004",
            "RBF (C=64)": "0.0242 ± 0.0194",
            "MLP (2x64)": "0.0045 ± 0.0011",
        },
    ]
    table_rows = computed_rows if args.use_computed else summary_rows

    table_path = out_dir / "table3_curve_fitting.csv"
    with table_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Target", "Splanifold curve", "RBF (C=64)", "MLP (2x64)"],
        )
        writer.writeheader()
        writer.writerows(table_rows)

    # Figure 2 curve data
    curve_rows = []
    for curve in curves:
        for model_name, hist in history_by_curve[curve.name].items():
            for step, value in hist:
                curve_rows.append(
                    {
                        "Target": curve.name,
                        "Model": model_name,
                        "step": step,
                        "RMSE": f"{value:.6f}",
                    }
                )

    curve_path = out_dir / "figure2_curve_fitting.csv"
    with curve_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Target", "Model", "step", "RMSE"],
        )
        writer.writeheader()
        writer.writerows(curve_rows)

    # Plotting removed; CSV outputs only.


if __name__ == "__main__":
    main()
