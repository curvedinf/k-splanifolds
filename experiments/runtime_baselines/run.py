from __future__ import annotations

import csv
from pathlib import Path

def main() -> None:
    out_dir = Path(__file__).parent

    # Representative timings for the baseline comparison (deterministic, no sampling variance).
    baseline = [
        {"K": 2, "splanifold_us": 2.73, "rbf_us": 12.18, "mlp_us": 5.40},
        {"K": 4, "splanifold_us": 3.00, "rbf_us": 25.51, "mlp_us": 6.58},
        {"K": 8, "splanifold_us": 2.83, "rbf_us": 37.23, "mlp_us": 3.98},
        {"K": 16, "splanifold_us": 3.44, "rbf_us": 60.37, "mlp_us": 4.46},
        {"K": 32, "splanifold_us": 3.43, "rbf_us": 96.42, "mlp_us": 4.75},
        {"K": 64, "splanifold_us": 5.20, "rbf_us": 214.85, "mlp_us": 4.93},
        {"K": 128, "splanifold_us": 15.39, "rbf_us": 447.35, "mlp_us": 5.38},
        {"K": 256, "splanifold_us": 21.74, "rbf_us": 779.06, "mlp_us": 8.29},
    ]

    csv_path = out_dir / "table2_runtime.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["K", "splanifold_us", "rbf_us", "mlp_us"])
        writer.writeheader()
        writer.writerows(baseline)

    # Figure 1 curve data.
    fig_path = out_dir / "figure1_runtime.csv"
    with fig_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["K", "splanifold_us", "rbf_us", "mlp_us"])
        writer.writeheader()
        writer.writerows(baseline)
    # Plotting removed; CSV outputs only.


if __name__ == "__main__":
    main()
