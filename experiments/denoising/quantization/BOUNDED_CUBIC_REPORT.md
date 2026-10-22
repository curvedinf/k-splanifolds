# Smoothly bounded cubic KS2: FP16 training and stress ablation

New variant trained from scratch for 1000 updates on each of three seeds. Earlier controls were reused, not retrained. Same 113K model and denoising protocol.

| Variant | Validation PSNR | Set12 PSNR |
|---|---:|---:|
| Previous bounded damping, unbounded cubic | 25.120752 | 26.711285 |
| Bounded damping and bounded cubic | 25.121055 ± 0.011647 | 26.711729 ± 0.011947 |

Set12 change: +0.000444 dB. Means ± sample SD over seeds. A small difference does not establish a quality improvement.

## Change

Replace the longitudinal basis input with 4*tanh(chart_mean/4), evaluated from the FP32 chart mean and rounded to FP16. The same bounded coordinate feeds the four Hermite terms and the longitudinal warped Chebyshev terms. Its magnitude cannot exceed four, and the cubic magnitude cannot exceed 64. This map is smooth, has slope one at zero and preserves ordering before finite-precision saturation. It deliberately changes extrapolation at large chart means.

Original chart coordinates still feed the coordinate warp, relative damping predicates and latent residual. Thus this bounds the polynomial input without globally normalizing or clipping the chart. The bound is fixed at four; no additional parameters or hyperparameter sweep were introduced. The previously tested bounded damping is retained.

## Stress results

All tested outputs, input gradients and geometry-parameter gradients were finite. There were 13 cases per checkpoint, 39 cases total: zero; random coordinate scales 1e-7, 1e-5, 0.01, 1, 10, 100, 1000, 10000 and 60000; all-positive 60000; all-negative 60000; and alternating +60000/-60000. Random samples are clipped to [-60000,60000] to stay inside FP16 input range with residual headroom. Each case contains 128×32 charts. Stress differentiation uses mean squared geometric correction, with FP32 loss reduction.

The prior unbounded-cubic variant failed at random scale100. The new variant passes that case and the larger cases. All inspected basis, polynomial, denominator, edge and geometric-output tensors stayed finite. Raw diagnostics include their maximum magnitudes. The tanh argument inside the coordinate warp is not among those inspected tensors; intermediate saturation there is not excluded by this test.

## Limits

This establishes finite behavior on the tested domain and checkpoints, not an unconditional bound on the whole network. Learned coefficient growth, dense projections, losses, optimizer states, input conversion outside FP16 range and residual addition can still overflow. At very large means the longitudinal tanh saturates, so its gradients vanish; stability comes with changed extrapolation. Near-zero predicate smoothness and general gradient underflow were not fully characterized. The original residual still carries chart coordinates, but half-precision rounding can erase small corrections to large values.

This is a CPU numerical experiment. FP32 master weights/optimizer, projections and small chart reductions/predicates remain; primary geometric arithmetic and edge accumulation use FP16. CPU contraction/tanh implementations can use wider internal arithmetic. No GPU speed claims.

## Reproduce

```bash
python -m pip install -r requirements.txt
python cubic_run.py
python cubic_checks.py
python cubic_report.py
```

Source, checkpoints, raw training/test/stress diagnostics and images are included. Original KS/data license terms apply.