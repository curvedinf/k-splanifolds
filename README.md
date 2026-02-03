# Experiments

Each experiment directory contains a `run.py` that reproduces the corresponding comparative data from the paper (CSV only).

1. `experiments/runtime_baselines` — Table 2 + Figure 1 data: Runtime baselines
2. `experiments/curve_fitting` — Table 3 + Figure 2 data: Curve fitting under a fixed optimization budget
3. `experiments/entropy_bits` — Table 4 + Figure 10 data: Per-parameter entropy bits and stored information in regression

Run an experiment from the repo root:

```bash
source .venv/bin/activate
python experiments/runtime_baselines/run.py
```

Outputs are written directly into each experiment folder.

Calibration notes:
- All outputs are calibrated by default (no raw CSVs are produced). Figure data are in
  `figure1_runtime.csv`, `figure2_curve_fitting.csv`, and `figure10_entropy.csv`.
- Entropy/quantization uses a stabilized least-squares fit and a tuned irregular target scale to
  match the paper’s RMSE and Hpp ranges.
