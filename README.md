# K-Splanifolds

This repo contains the K‑Splanifolds paper plus the code and data used to generate its comparative experiments.

**KS3** generalizes the selected spine FFN to configurable subspace ranks: learned input/output projections,
a bounded Hermite cubic plus two warped Chebyshev terms, a coordinate bypass,
and ReLU². At width 96 with 368 rank-1 subspaces it has **73,600 parameters per FFN**.

- [ks3.py](experiments/common/ks3.py) — standalone, readable PyTorch implementation with autograd.
- [ks3_int8.py](experiments/common/ks3_int8.py) — integrated W8A8,
  GS32 with FP16 quantization scales, fused Horner evaluation and tensor parallelism; FP16 nonlinearities by default,
  with FP32 RWKV recurrent state.
- [Measurements and export instructions](experiments/rwkv7_ks2/reports/2026-10-23-rank1-int8-tp/REPORT.md)
  — historical results originally labeled rank-1 KS2; the KS3 rename does not change the algorithm.

```python
import torch
from experiments.common.ks3 import KS3

ffn = KS3(in_features=96, num_subspaces=368)
y = ffn(torch.randn(2, 64, 96))  # (batch, time, model width)
```

The Torch reference uses floating-point weights and supports training. The GPU
export is quantized inference. Normalization, attention and the outer residual
connection are supplied by the enclosing model.

Convert a trained cell to fused INT8 inference (PyTorch + Triton on GPU):

```python
from experiments.common.ks3_int8 import KS3Int8

# Load the trained weights into ffn before packing them.
quantized_ffn = KS3Int8(ffn.cuda().eval())
with torch.inference_mode():
    y = quantized_ffn(torch.randn(2, 64, 96, device="cuda"))
```

Both implementations accept arbitrary positive integer input/output widths,
subspace counts and subspace ranks. For example:

```python
ffn = KS3(in_features=129, out_features=65, num_subspaces=37, subspace_rank=2)
```

Each chart's spine uses its mean input coordinate and learns six coefficients
per output coordinate. Parameter count is
`num_subspaces * (subspace_rank * (in_features + out_features + 6) + 2)`.
Rank 1 keeps the fused input-GEMM/geometry kernel; higher ranks add a chart-mean
reduction and a fused geometry/requantization kernel. The coordinate clamp,
bounded cubic coordinate and tanh warp are preserved. KS3 has no edge terms,
so KS2's edge-energy damping is not part of this spine-only algorithm.

All INT8 matrices, activations, embeddings and coefficient rows use GS32 with
FP16 quantization scales. Rows shorter than 32 (including six-coefficient rows)
use partial groups. Scales are rounded before encoding and promoted to FP32 for
arithmetic, with bounds `[2**-24, 65504]`; codes saturate to `[-127,127]`.
Warp scalars, optimizer masters/gradients and recurrent state remain FP32.

TP keeps whole charts and whole quantization groups together. The default
368-chart rank-1 model uses `[96, 96, 96, 80]` features at TP4. Other positive TP
world sizes are accepted; tiny layouts can leave ranks without charts, and those
ranks still join reductions. `KS3Int8` is standalone; full RWKV conversion uses
the repository's attention adapter. FP32 and INT8 nonlinear modes remain available.
The GS32 export format is version 2; older mixed-group exports must be regenerated
from their original floating-point checkpoints.

The same file also implements fused first-order backward for INT8 FFN training:

```python
qat_ffn = KS3Int8(ffn.cuda(), trainable=True)
optimizer = torch.optim.AdamW(qat_ffn.parameters(), lr=1e-3)
x = torch.randn(2, 64, 96, device="cuda", requires_grad=True)
optimizer.zero_grad(set_to_none=True)
loss = qat_ffn(x).square().mean()  # Replace with the model's training objective.
loss.backward()
optimizer.step()
frozen_ffn = qat_ffn.to_inference()
```

Training keeps FP32 master parameters and gradients, with the same fused W8A8
forward and FP16 nonlinear default. The backward fuses the output-projection
gradient with the KS3 geometry derivative; its GEMMs dequantize saved INT8 operands
inside the kernel and accumulate in FP32. Quantization scales are detached and
rounding uses an identity straight-through estimator. Nonlinear derivatives use
smooth FP32 KS3 geometry, including for the FP16 and INT8 forward modes.
TP sums input gradients and keeps parameter gradients local to each shard.
The training API covers the FFN; full RWKV conversion remains inference-only.

For an existing model, `fuse_training(cell)` in `ks3_int8.py` enables the same
kernel on a Torch KS3 cell while preserving its parameter objects, optimizer
groups and checkpoint keys. It accepts FP32/FP16/BF16 inputs, supports autocast,
activation checkpointing and `cell.compile(fullgraph=True, dynamic=False)`, and
keeps master parameters FP32. The [nanoGPT integration](experiments/nanogpt_ks2/README.md)
uses this interface with `--models mlp ks3 --device cuda`.

For that nanoGPT harness, `--ks3-graph` enables captured FFN training replay;
other shapes and evaluation use the ordinary fused path. The kernel batches
weight/activation packing and backward calculations, and the harness fuses its
existing KS3 geometry Adam updates. See the [matched MLP control comparison](experiments/nanogpt_ks2/reports/2026-10-23-ks3-control/REPORT.md)
for measured training timing and loss.

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

