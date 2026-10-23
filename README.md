# K-Splanifolds

This repo contains the K‑Splanifolds paper plus the code and data used to generate its comparative experiments.

**KS3** is the selected rank-1 spine FFN: learned input/output projections,
a bounded Hermite cubic plus two warped Chebyshev terms, a coordinate bypass,
and ReLU². At width 96 with 368 subspaces it has **73,600 parameters per FFN**.

- [ks3.py](ks3.py) — standalone, readable PyTorch implementation with autograd.
- [KS3 GPU implementation](experiments/rwkv7_ks2/ks3_gpu.py) — integrated W8A8,
  fused Horner evaluation and tensor parallelism; FP16 nonlinearities by default,
  with FP32 RWKV recurrent state.
- [Measurements and export instructions](experiments/rwkv7_ks2/reports/2026-10-23-rank1-int8-tp/REPORT.md)
  — historical results originally labeled rank-1 KS2; the KS3 rename does not change the algorithm.

```python
import torch
from ks3 import KS3

ffn = KS3(in_features=96, num_subspaces=368)
y = ffn(torch.randn(2, 64, 96))  # (batch, time, model width)
```

The Torch reference uses floating-point weights and supports training. The GPU
export is quantized inference. Normalization, attention and the outer residual
connection are supplied by the enclosing model.

Key files:
- `k-splanifolds.pdf` — the paper ([reference link](https://doi.org/10.5281/zenodo.18673034)).
- `k-splanifolds-2D-to-3D-toy.html` — interactive 2D→3D toy visualization.
- `k-splanifolds-3D-to-3D-visualization.html` — 3D→3D visualization.
- `k-splanifolds.mp4` — short demo video.
- `setup_venv.sh` — creates a local `.venv` and installs dependencies.

Key experiment code:
- [`experiments/rwkv7_ks2`](experiments/rwkv7_ks2) — FP32 RWKV-7 controller/FFN ablations, width sweeps, update-scale diagnostics, and recorded CPU loss/speed/memory results.
- [`experiments/nanogpt_ks2`](experiments/nanogpt_ks2) — FP32 modded-nanoGPT controls, fused backward, memory/inference measurements, and INT8/Q4_K/IQ2_XXS/TQ1 quantization and QAT.
- [`experiments/denoising`](experiments/denoising) — MLP/KS1/KS2 controls, 3,000-step ternary QAT, and historical FP16/FP8/INT8 ablations.
- `experiments/common/ks2.py` — reference KS2 map and projected cell.
- `experiments/common/ks2_stabilized.py` — trainable KS2 with bounded cubics and energy-based damping.
- [`experiments/common/ks2_cpu`](experiments/common/ks2_cpu) — fused AVX-512 FP32 Horner and GS128 INT8 VNNI exports.
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

