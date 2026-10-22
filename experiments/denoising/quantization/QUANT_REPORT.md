# Three GPU-oriented KS2 quantization ablations

CPU numerical emulation only: no GPU was available. These tests measure frozen-checkpoint quality, not GPU throughput, training stability, or scaling with model size. All three existing 113K rank8/32-chart checkpoints were evaluated without retraining. Dense projections remain FP32 to isolate inner geometry.

## Combined results

| Configuration | Validation PSNR | Set12 PSNR | Set12 change from FP32 |
|---|---:|---:|---:|
| fp32 / all | 25.111383 | 26.696575 | +0.000000 |
| fp16 / all | 25.111397 | 26.696577 | +0.000002 |
| fp16 / safe | 25.111391 | 26.696576 | +0.000000 |
| fp8 / all | 25.110546 | 26.695378 | -0.001198 |
| int8 / all | 25.111392 | 26.696703 | +0.000128 |

Means across three checkpoints. These tiny changes do not establish a quality improvement. Raw per-seed/per-image results and sample standard deviations are included.

## 1. FP16 inner arithmetic

Projected coordinates, effective coefficients, tanh warps, polynomial intermediates, normalized coordinate weights, damping denominators, edge products, edge division, and residual addition are rounded to FP16. Edge accumulation explicitly rounds after each of eight additions. The dense output projection receives FP32-converted geometry output. PyTorch CPU reductions/tanh may compute internally at higher precision before rounding; this is not an exact simulation of every possible GPU instruction sequence, fusion, FMA or subnormal handling.

The safe variant retains FP16 coordinates, coefficients, warps and polynomial evaluation, but restores FP32 coordinate reduction/normalization, damping construction, edge product/division, accumulation and residual addition. Thus this comparison isolates a bundle of safeguards, not which individual safeguard is necessary.

Both retain effectively unchanged benchmark PSNR. Nevertheless the broadly FP16 path develops infinite weights/squared weights/denominators. Infinite denominators suppress edge contributions to zero and do not make the final outputs nonfinite here. This behavior could become more important during training or on a different input distribution.

| Seed | Near-zero fallback changes / 131072 charts | Nonfinite normalized weights | Nonfinite denominator entries |
|---|---:|---:|---:|
| 11 | 0 | 0 | 0.0911% |
| 22 | 7 | 4 | 0.1081% |
| 33 | 1 | 0 | 0.1132% |

## 2. FP8 coefficient storage

E4M3FN storage with a static power-of-two scale per chart/family. Quantized effective coefficients are expanded to FP32 for geometry arithmetic. Separate validation ablations cover longitudinal, coordinate, damping and warp families, followed by their combination. FP8 E5M2 and FP8 internal arithmetic were not tested.

## 3. INT8 coefficient storage, GS128

Symmetric INT8 (-127..127), static absmax scales and round-to-nearest-even. Quantized values are expanded into FP32 arithmetic. Groups stay within a chart and family: longitudinal 432 values uses groups 128/128/128/48; coordinate 256 uses 128/128; damping 128 uses one group; warp 4 uses a short group. Padding does not add stored model values. Scales themselves remain FP32.

The initial per-chart INT8 results were superseded by GS128 after the user specified group size. The old script/results are preserved only for provenance; the main tables contain GS128.

## Family sensitivity (validation only)

| Format | Quantized family | Validation change dB | Maximum normalized-output difference |
|---|---|---:|---:|
| fp8 | longitudinal | -0.000259 | 0.009225 |
| fp8 | coordinate | -0.000288 | 0.028893 |
| fp8 | damping | +0.000007 | 0.000379 |
| fp8 | warp | -0.000305 | 0.009801 |
| fp8 | all | -0.000837 | 0.031117 |
| int8 | longitudinal | -0.000002 | 0.005656 |
| int8 | coordinate | +0.000034 | 0.007825 |
| int8 | damping | +0.000000 | 0.000041 |
| int8 | warp | -0.000020 | 0.001980 |
| int8 | all | +0.000009 | 0.009635 |

All four coefficient families tolerate the tested storage quantization on this benchmark. Effective coefficients are the precomputed Horner coefficients, positive damping values and transformed warp scales/shifts used by the frozen inference kernel, not the original trainable parameter encoding. Quantization is performed after that precomputation. These storage ablations do not quantize intermediate activations to FP8/INT8.

## Practical interpretation

The results support testing FP16 geometry and low-precision coefficient storage in an actual GPU kernel. They do not show that all internal computation can run in INT8 or FP8. Retaining FP32 for normalization and damping is justified by observed FP16 overflow rather than a demonstrated PSNR benefit. Whether those safeguards matter for gradients requires a training experiment; none was performed here. No throughput claims follow from CPU emulation.

## Reproduce

```bash
python -m pip install -r requirements.txt
python quant_ablation.py
python quant_diagnostics.py
python quant_report.py
```

The archive includes scripts, raw results, existing checkpoints and dataset images. No new training was run. Original KS/data license terms apply.