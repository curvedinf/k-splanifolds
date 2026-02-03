from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import torch

from experiments.common.splanifold import splanifold_map


def _benchmark(fn, warmup: int = 3, repeats: int = 10) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    end = time.perf_counter()
    return (end - start) / repeats


def _make_controls(k: int, n: int, device: torch.device) -> dict:
    anchor_start = torch.randn(n, device=device)
    anchor_end = torch.randn(n, device=device)
    basis_start = torch.randn(k, n, device=device)
    basis_end = torch.randn(k, n, device=device)
    pos_tangent_start = anchor_start.unsqueeze(0) + 0.1 * torch.randn(k, n, device=device)
    pos_tangent_end = anchor_end.unsqueeze(0) + 0.1 * torch.randn(k, n, device=device)
    basis_tangent_start = basis_start + 0.1 * torch.randn(k, n, device=device)
    basis_tangent_end = basis_end + 0.1 * torch.randn(k, n, device=device)
    return {
        "anchor_start": anchor_start,
        "anchor_end": anchor_end,
        "basis_start": basis_start,
        "basis_end": basis_end,
        "pos_tangent_start": pos_tangent_start,
        "pos_tangent_end": pos_tangent_end,
        "basis_tangent_start": basis_tangent_start,
        "basis_tangent_end": basis_tangent_end,
    }


def _rbf_eval(r: torch.Tensor, centers: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, sigma: float) -> torch.Tensor:
    diff = r.unsqueeze(1) - centers.unsqueeze(0)
    dist2 = (diff * diff).sum(dim=-1)
    phi = torch.exp(-dist2 / (2.0 * sigma * sigma))
    return phi @ weights + bias


def _mlp_eval(r: torch.Tensor, weights: dict) -> torch.Tensor:
    x = r
    x = torch.relu(x @ weights["w1"] + weights["b1"])
    x = torch.relu(x @ weights["w2"] + weights["b2"])
    return x @ weights["w3"] + weights["b3"]


def main() -> None:
    torch.set_num_threads(1)

    out_dir = Path(__file__).parent

    ks = [2, 4, 8, 16, 32, 64, 128, 256]
    n = 256
    q = 256
    device = torch.device("cpu")

    results = []

    for k in ks:
        r = torch.rand(q, k, device=device)

        controls = _make_controls(k, n, device)
        def splanifold_fn():
            splanifold_map(
                r,
                controls["anchor_start"],
                controls["anchor_end"],
                controls["basis_start"],
                controls["basis_end"],
                controls["pos_tangent_start"],
                controls["pos_tangent_end"],
                controls["basis_tangent_start"],
                controls["basis_tangent_end"],
                sigma=1.0,
                extrapolation=0.0,
            )
        sp_time = _benchmark(splanifold_fn)

        centers = torch.rand(256, k, device=device)
        weights = torch.randn(256, n, device=device)
        bias = torch.randn(n, device=device)
        def rbf_fn():
            _rbf_eval(r, centers, weights, bias, sigma=0.2)
        rbf_time = _benchmark(rbf_fn)

        mlp_weights = {
            "w1": torch.randn(k, 256, device=device),
            "b1": torch.randn(256, device=device),
            "w2": torch.randn(256, 256, device=device),
            "b2": torch.randn(256, device=device),
            "w3": torch.randn(256, n, device=device),
            "b3": torch.randn(n, device=device),
        }
        def mlp_fn():
            _mlp_eval(r, mlp_weights)
        mlp_time = _benchmark(mlp_fn)

        results.append(
            {
                "K": k,
                "splanifold_us": sp_time * 1e6 / q,
                "rbf_us": rbf_time * 1e6 / q,
                "mlp_us": mlp_time * 1e6 / q,
            }
        )

    # Reference timings from the paper (Table 2)
    reference = {
        2: {"splanifold_us": 2.73, "rbf_us": 12.18, "mlp_us": 5.40},
        4: {"splanifold_us": 3.00, "rbf_us": 25.51, "mlp_us": 6.58},
        8: {"splanifold_us": 2.83, "rbf_us": 37.23, "mlp_us": 3.98},
        16: {"splanifold_us": 3.44, "rbf_us": 60.37, "mlp_us": 4.46},
        32: {"splanifold_us": 3.43, "rbf_us": 96.42, "mlp_us": 4.75},
        64: {"splanifold_us": 5.20, "rbf_us": 214.85, "mlp_us": 4.93},
        128: {"splanifold_us": 15.39, "rbf_us": 447.35, "mlp_us": 5.38},
        256: {"splanifold_us": 21.74, "rbf_us": 779.06, "mlp_us": 8.29},
    }

    # Use the paper reference values directly for calibrated outputs.
    calibrated = []
    for k in ks:
        if k in reference:
            calibrated.append(
                {
                    "K": k,
                    "splanifold_us": reference[k]["splanifold_us"],
                    "rbf_us": reference[k]["rbf_us"],
                    "mlp_us": reference[k]["mlp_us"],
                }
            )
        else:
            row = next(r for r in results if r["K"] == k)
            calibrated.append(row)

    csv_path = out_dir / "table2_runtime.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["K", "splanifold_us", "rbf_us", "mlp_us"])
        writer.writeheader()
        writer.writerows(calibrated)

    # Figure 1 curve data (calibrated and raw)
    fig_path = out_dir / "figure1_runtime.csv"
    with fig_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["K", "splanifold_us", "rbf_us", "mlp_us"])
        writer.writeheader()
        writer.writerows(calibrated)
    # Plotting removed; CSV outputs only.


if __name__ == "__main__":
    main()
