# K-Splanifolds

This repo contains the K‑Splanifolds paper plus the code and data used to generate its comparative experiments.

Key files:
- `k-splanifolds.pdf` — the paper ([reference link](https://doi.org/10.5281/zenodo.18673034)).
- `k-splanifolds-2D-to-3D-toy.html` — interactive 2D→3D toy visualization.
- `k-splanifolds-3D-to-3D-visualization.html` — 3D→3D visualization.
- `k-splanifolds.mp4` — short demo video.
- `setup_venv.sh` — creates a local `.venv` and installs dependencies.

Key experiment code:
- `experiments/common/splanifold.py` — core splanifold map and utilities used by the experiments.
- `experiments/runtime_baselines/run.py` — runtime baselines (Table 2 + Figure 1 data).
- `experiments/curve_fitting/run.py` — curve fitting under a fixed optimization budget (Table 3 + Figure 2 data).
- `experiments/entropy_bits/run.py` — entropy-coded bits per parameter in regression (Table 4 + Figure 10 data).

## Setup & Running Experiments

Setup:
```bash
./setup_venv.sh
```

Running:
```bash
source .venv/bin/activate
python experiments/runtime_baselines/run.py
```
