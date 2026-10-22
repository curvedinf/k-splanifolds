# FP16 KS2 with bounded damping: training ablation

New stability math was trained from scratch; this was not a weight conversion. Three seeds, 1000 updates each, same 113K model and denoising protocol. CPU only; no GPU throughput claims.

| Variant | Successful updates per seed | Validation patch PSNR | Set12 whole-image PSNR |
|---|---|---:|---:|
| Earlier FP32 model, unchanged math | 1000 / 1000 / 1000 | 25.111384 | 26.696575 |
| Original damping in FP16 inner arithmetic | 0 / 0 / 0 | Failed first backward pass | Not evaluated |
| Bounded damping in FP16 inner arithmetic | 1000 / 1000 / 1000 | 25.120752 ± 0.011726 | 26.711285 ± 0.011979 |

The stable variant is +0.014710 dB relative to the historical FP32 reference on Set12. This small difference is not evidence of a general quality improvement. Its main result is completing training with finite gradients while the new original-math FP16 control fails.

## Math changes

1. Normalize chart coordinates by maximum absolute coordinate in FP32; use one as divisor for the all-zero chart. This scale is used for predicates only, not for the learned polynomial input.
2. Coordinate importance is normalized coordinate squared divided by the sum of squared normalized coordinates. Use zero importance for an all-zero chart.
3. Relative distance is squared normalized deviation from the mean divided by that same squared deviation plus mean squared normalized coordinate. Use zero for the all-zero case.
4. Both predicates lie between zero and one. Each damping strength is eight times a sigmoid, with a fixed logit offset so the initial strength matches the original softplus(-4).
5. Construct the denominator from one plus the two strength/predicate products. It is bounded between one and seventeen for finite input. No coordinate is divided by a cancelling signed sum.
6. Apply attenuation to the longitudinal edge factor before multiplying it by the coordinate edge factor.

The optional cancellation gate was deliberately omitted in this first ablation. The original near-zero uniform-weight fallback and signed-sum sensitivity are removed from damping. Hermite/Chebyshev functions, chart count/rank, dense projections, parameter count and learning schedule are unchanged. This is an architectural change; it is not algebraically equivalent to the previous model.

## Precision boundaries

Both new variants keep FP32 trainable master parameters, optimizer state, loss, projections and a small chart reduction. The stable variant additionally evaluates bounded predicates in FP32 and casts them to FP16. Basis polynomials, tanh warp outputs, coefficient bmm inputs/outputs, denominator construction, edge multiplication/division and latent residual are FP16. Edge accumulation explicitly rounds after each of eight additions. There is no dynamic loss scaler. CPU bmm/reduction/tanh may accumulate or evaluate internally at higher precision before rounding, so results are not an exact GPU instruction simulation.

## Observed failures and bounds

The original FP16 control generated nonfinite den_weight gradients at step one in every seed and stopped before an optimizer update. Its validation diagnostic batch had 104–240 nonfinite denominator entries out of 262144. All three trained stable variants had zero nonfinite entries in that diagnostic batch; observed denominators ranged from 1 to approximately 1.033. The cap of 17 was not approached.

Every completed training step checked finite loss and every parameter gradient. All validation and Set12 scores were finite. Training clocks include gradient checks and are CPU-only, so they should not be compared as kernel throughput results.

## Stress test and limits

Seed 11 was additionally tested on zero coordinates and random coordinates scaled by 1e-7, 1e-5, 0.01, 1 and 10: outputs and input gradients stayed finite. At scale 100, the unchanged unbounded cubic polynomial path produced nonfinite outputs/gradients, while the damping denominator remained finite. Bounded damping therefore fixes the observed damping failure, not every possible overflow or underflow in the network. Gradient underflow was not comprehensively measured. The normalized predicates are direction-dependent near an all-zero chart; the explicit zero case is finite but this is not a global smoothness guarantee.

## Reproduce

```bash
python -m pip install -r requirements.txt
python stable_run.py
python stable_checks.py
python stable_report.py
```

The archive includes new checkpoints, source, raw training/test/diagnostic results, dataset images and historical metric references. Earlier controls were not retrained, apart from the explicitly new FP16 training control. Original KS/data license terms apply.