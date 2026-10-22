# INT8-aware KS2 training from scratch

Two new variants trained from scratch for 1000 steps on each of three seeds. Quantized operations were active from step one; no floating-point warmup or pretrained initialization. The earlier FP16-geometry training is reused as the baseline.

| Variant | Completed updates per seed | Validation PSNR | Set12 PSNR |
|---|---|---:|---:|
| Earlier FP16 geometry | 1000 / 1000 / 1000 | 25.121055 | 26.711729 ± 0.011947 |
| INT8-aware contractions + edges | 1000 / 1000 / 1000 | 25.121769 | 26.713085 ± 0.012069 |
| INT8-aware contractions + edges + powers | 1000 / 1000 / 1000 | 25.121931 | 26.713632 ± 0.011412 |

Set12 mean ± sample SD across three checkpoint means. Best validation checkpoint within each run is used for test evaluation. All six new runs completed 1000 optimizer updates with finite losses and parameter gradients. Small quality differences are not evidence of a general improvement.

## What was actually trained

Forward passes execute INT8-quantized basis/coefficient contractions and edge products with INT32 accumulation. The aggressive variant also executes integer square/cube products with requantization between powers. GS128 weight scales are recomputed from current FP32 master weights on each forward pass. There is no inference-only post-training substitution: these operations are active while the weights are learned.

Backward uses a floating-point straight-through surrogate because integer rounding has no useful ordinary gradient. Its forward value is replaced by the exact integer-path value, and its gradient follows quantize/dequantize surrogate operands. Activation gradients are suppressed outside the quantization clipping range. Scale estimation is detached from the gradient graph. Weight masters and Adam state remain FP32; this is quantization-aware training, not an all-integer optimizer or integer backward pass.

Dense projections remain FP32. Tanh, damping/division, rescaling and residual addition remain floating point; surrounding geometry uses the previous FP16 implementation and its FP32 predicate reductions. CPU integer operands are widened to INT32 for multiplication; actual GPU instruction throughput is not measured.

## Adaptive activation scales

Longitudinal basis, attenuated longitudinal edges and coordinate edges each have a per-chart activation range. During training, ranges track batch maxima with a 0.95 exponential moving average and 20% headroom. The first use initializes the range. Evaluation freezes these buffers; validation and test images never update them. Coordinate Chebyshev basis and polynomial powers use their known bounds. The saved checkpoint includes the learned weights and running ranges.

## Verification

Every training step checked loss and all existing parameter gradients for finiteness before clipping and updating. Additional checks on all six trained checkpoints verified:

- Exact output equality between QAT evaluation and the standalone integer inference implementation using exported weights and ranges.
- Exact output equality with gradients enabled versus disabled.
- Finite, nonzero gradients in every trainable parameter group on an independent random batch.
- Evaluation does not change the saved activation ranges.

These checks establish that the model was learning through the quantized stages and that the deployed integer path matches the trained forward path. They do not establish stability for all tasks, ranks, model sizes or training durations. Gradient underflow and sensitivity to other scaling policies were not exhaustively studied.

## Protocol

Same 360/40 Train400 split,120000 augmented clean patches,4096 validation patches,13×13 grayscale patches,Gaussian noise25/255,batch64,Adam LR0.0003,gradient clip1,seeds11/22/33. Exactly1000 updates per run. Validation at1/250/500/750/1000. Set12 uses the original fixed noise and overlap reconstruction. No learning-rate tuning or extra training trials. CPU only; training clocks include checks and duplicate surrogate computation and should not be read as GPU performance measurements.

## Reproduce

```bash
python -m pip install -r requirements.txt
python qat_run.py
python qat_checks.py
python qat_report.py
```

Scripts, new checkpoints, training/test logs, export/gradient checks, images and baseline references are included. Original KS/data license terms apply.