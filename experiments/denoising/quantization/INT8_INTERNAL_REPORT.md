# INT8 internal arithmetic in bounded KS2

Frozen-checkpoint inference ablation, three seeds. CPU integer arithmetic; not a GPU kernel or speed benchmark. Dense projections remain FP32. This goes beyond low-bit weight storage: selected inner multiplications and reductions execute as integer operations.

| Variant | Validation PSNR | Set12 PSNR | Set12 change |
|---|---:|---:|---:|
| FP16 geometry reference | 25.121055 | 26.711729 | +0.000000 |
| INT8 basis/coefficient contractions | 25.121002 | 26.710927 | -0.000802 |
| INT8 edge products + INT32 edge sum | 25.121087 | 26.711480 | -0.000249 |
| Both integer stages | 25.120988 | 26.710891 | -0.000838 |
| Both stages + INT8 squares/cubes | 25.119445 | 26.710883 | -0.000846 |

Quality changes are measured relative to the same bounded-cubic FP16 model. Tiny increases are not evidence of improvement. Test scores are means across three checkpoint means; raw per-image results and seed standard deviations are included.

## Integer contractions

Longitudinal basis values are quantized to INT8 using one static activation scale per chart. The 432 longitudinal coefficients in each chart use GS128 blocks (128/128/128/48). Each block performs a six-term INT32 dot product using INT8-quantized operands widened to INT32 on CPU. Each block result is rescaled and the block results are combined in FP32, then rounded to FP16. This represents grouped quantized contractions rather than a single scale shared by all coefficients.

Coordinate Chebyshev basis values use the known [-1,1] range. Their 256 coefficients per chart use two GS128 groups. Each group covers four input coordinates; each input/output coefficient contraction is a four-term INT32 dot product, then rescaled to FP16.

## Integer edge products and reduction

Damping and the division of longitudinal edge factors by their denominators remain floating point. The attenuated factor and coordinate-edge factor are quantized to INT8 with two fixed per-chart scales. Their products and the sum over eight input coordinates execute in INT32, followed by one floating-point rescaling. Products within a chart share scales so the integer sum is meaningful without per-edge rescaling.

The largest absolute eight-term sum is bounded by 8×127×127=129032, comfortably inside INT32. A six-term contraction is bounded by 96774. These limits protect these specific integer accumulators; the entire network is not integer arithmetic.

## Aggressive power evaluation

The last configuration also computes square/cube products using INT8 operands widened to INT32, with round-and-requantize between multiplications. Longitudinal t uses the fixed bound4; tanh outputs use bound1. The resulting powers are converted back to FP16 for the Hermite/Chebyshev linear combinations. Tanh itself, damping predicates and division remain floating point.

The fixed longitudinal square and cube quantization steps are 16/127 and 64/127. This is intentionally aggressive near zero: small higher-order terms can round to zero. It probes whether that internal simplification damages this task, not whether it exactly reproduces the previous function.

## Calibration and saturation

Activation scales were calibrated on 1024 augmented patches drawn only from the training split, with independent fixed noise. Calibration is performed once per checkpoint in reference FP16 mode; all modes reuse those scales. Validation and test data do not update them. Coefficient scales use only the checkpoint weights. All scales remain FP32. Activation values exceeding calibrated ranges saturate to ±127, so numerical finiteness does not imply accurate extrapolation.

Validation saturation rates for the calibrated activations are recorded below. Known-range polynomial-power quantization is separate.

| Seed | Mode | Tensor | Saturated fraction |
|---|---|---|---:|
| 11 | contracts | basis | 0.020727% |
| 11 | edges | atten | 0.003779% |
| 11 | edges | ex | 0.000346% |
| 11 | combined | basis | 0.020727% |
| 11 | combined | atten | 0.002575% |
| 11 | combined | ex | 0.012088% |
| 11 | powers_all | basis | 0.019836% |
| 11 | powers_all | atten | 0.005031% |
| 11 | powers_all | ex | 0.020802% |
| 22 | contracts | basis | 0.023905% |
| 22 | edges | atten | 0.003541% |
| 22 | edges | ex | 0.000775% |
| 22 | combined | basis | 0.023905% |
| 22 | combined | atten | 0.007606% |
| 22 | combined | ex | 0.007820% |
| 22 | powers_all | basis | 0.027974% |
| 22 | powers_all | atten | 0.045192% |
| 22 | powers_all | ex | 0.013316% |
| 33 | contracts | basis | 0.023270% |
| 33 | edges | atten | 0.003445% |
| 33 | edges | ex | 0.001955% |
| 33 | combined | basis | 0.023270% |
| 33 | combined | atten | 0.004053% |
| 33 | combined | ex | 0.007772% |
| 33 | powers_all | basis | 0.026194% |
| 33 | powers_all | atten | 0.014317% |
| 33 | powers_all | ex | 0.018418% |

## Scope and limits

All validation and reconstructed test outputs were finite. No quantization-aware training or gradient experiment was performed. These results do not establish trainability with INT8 arithmetic, suitability for larger ranks/models, or GPU acceleration. Integer operands are widened to INT32 for CPU execution; actual GPU INT8 dot-product instructions, layout, conversion cost and throughput are not measured. Small contraction sizes and groupwise rescaling can affect practical GPU efficiency. No claim is made that this hybrid is an all-INT8 network.

## Reproduce

```bash
python -m pip install -r requirements.txt
python int8_internal_run.py
python int8_internal_report.py
```

Scripts, checkpoint weights, dataset images, raw metrics and calibration/clipping details are included. Original KS/data license terms apply.