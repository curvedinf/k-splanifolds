# KS2 · stabilized CPU kernels

Train with [`KS2StabilizedCell`](../ks2_stabilized.py); export with **`KS2Horner`** or **`KS2VNNI`**.
Both fuse chart geometry and the residual into one OpenMP call. Projections remain FP32 PyTorch GEMMs.

## Techniques versus reference KS2

| Component | Reference `KS2Cell` | Stabilized / native implementation |
|---|---|---|
| Longitudinal coordinate | Unbounded chart mean | Smooth tanh cap at ±4; cubic bounded by 64 |
| Importance damping | Coordinate / signed chart sum, squared | Share of chart energy; no signed cancellation singularity |
| Distance damping | Absolute deviation from mean | Squared deviation / (squared deviation + mean energy) |
| Damping strengths | Unbounded softplus | Sigmoid bounded to 0–8; same initial strength |
| Near-zero energy | Signed-sum fallback | Positive denominator floors / guarded zero division |
| Geometry precision | Follows tensor operations | FP32 geometry, including under autocast in the trainable implementation |
| Extreme coordinates | No explicit cap | Saturate latent coordinates at ±65504 |
| Polynomial evaluation | Explicit bases and contractions | Horner: precombined coefficients and FMA |
| SIMD layout | Framework contractions | Horner: 16 patches/vector; VNNI: 16 longitudinal or 8 edge outputs/vector |
| Quantized contractions | None | VNNI: GS128 signed-byte weights, INT32 accumulation, unsigned-offset corrections |
| Deployment overhead | Repeated tensor operations | Frozen prepacking, empty-block pruning, unrolling, one geometry call |
| Nonlinear functions | Framework tanh and division | Native: bounded rational tanh; reciprocal estimate + Newton refinement |
| Intermediate rounding | Depends on dtype | Native: no repeated FP16 round trips |

**These are modified KS2 functions, not numerically identical replacements for `ks2.py`.**
The trainable stabilized classes retain its parameter layout, but change the geometry.
The native exports add quantization and numerical approximations. `KS2Horner` folds GS128-quantized
weight values into **FP32** coefficients; it does not execute INT8 arithmetic. `KS2VNNI` retains
integer coefficient contractions and edge products, with floating nonlinearities and projections.

## Use

Linux x86-64, GCC/OpenMP and PyTorch. Both exports require AVX-512F/BW/DQ/VL and FMA;
VNNI additionally requires AVX-512 VNNI. The wrapper checks CPU capabilities before native calls.
Build locally with explicit ISA flags; no `-ffast-math` or external vector-math dependency.

```bash
python -m experiments.common.ks2_cpu.build
python -m unittest experiments.common.ks2_cpu.test_kernels -v
```

```python
import torch
from experiments.common.ks2_stabilized import KS2StabilizedCell
from experiments.common.ks2_cpu import KS2Horner, KS2VNNI

model = KS2StabilizedCell(169, num_subspaces=32, subspace_rank=8, geo_steps=1)
# Train with ordinary backprop, then calibrate on representative input batches.
model.eval().calibrate(calibration_batches)
horner = KS2Horner(model, threads=8)
integer = KS2VNNI(model, threads=8)

with torch.inference_mode():
    prediction = horner(patches)  # CPU float32 [batch, 169]; try batches of 512.
```

`KS2Stabilized` is the direct map; `KS2StabilizedCell` adds projections and residual steps.
Both support arbitrary ranks and backprop. Native exports require **rank 8, one geometry step**.
They are frozen snapshots: rebuild after training changes. Horner does not require calibration;
VNNI requires positive `range_basis`, `range_atten` and `range_ex` buffers. Calibration uses
per-chart maxima with 20% headroom by default; quality must be checked on held-out data.

Constructors also accept a state dict or `torch.load(..., weights_only=True)` checkpoint containing
`state_dict`. The prior packed `qat_combined` denoising layout is supported directly. Bare state
dicts must describe a one-step stabilized cell: step count is not encoded in tensor shapes.
Original KS2 checkpoint weights can initialize the stabilized classes, but their predictions change.

## Measured denoising performance

EPYC 9V74, 8 threads, three checkpoints × three Set12 passes; no retraining.

| Export | Batch 128 · ms/image | Batch 512 · ms/image | PSNR · dB |
|---|---:|---:|---:|
| Previous fused INT8 VNNI | 100.0 | 78.3 | 26.71309 |
| Optimized INT8 VNNI | 72.6 | 51.4 | 26.71306 |
| Stabilized FP32 AVX-512 Horner | 54.0 | 35.8 | 26.71494 |

Timings include normalization, model calls and concatenation; exclude unfold/fold, compilation
and prepacking. Set12: σ=25/255, 13×13 patches, stride 3, fixed noise seeds, overlap averaging.
These are the original experiment measurements, not a fresh benchmark of this packaging.
The port matches VNNI outputs exactly and Horner within 2.4e-7 across three saved checkpoints.

Stability checks cover zero, tiny values, cancellation and inputs up to 1e30. The exported geometry
is bounded for the tested finite parameters; this does not promise faithful extreme extrapolation,
finite projections near FP32 overflow, or stability for arbitrary parameter values. Tests also cover
trainable forward/backward, calibration, autocast, SIMD tails and invalid export configurations.
