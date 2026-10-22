# Historical denoising precision ablations

This directory preserves the original CPU numerical experiments, model implementations, raw results, splits, and reports that led to stabilized KS2. They are separate from the current all-weight ternary QAT runner at [`../train_tq1.py`](../train_tq1.py). The historical experiments mostly retain FP32 dense projections.

| Study | Report | Runner | Inputs |
|---|---|---|---|
| FP16 inner arithmetic, FP8 / INT8 coefficient storage | [Quantization](QUANT_REPORT.md) | `quant_ablation.py` | `rank_results/R8S32_seed{11,22,33}.pt` |
| Bounded damping with FP16 training | [FP16 stability](STABLE_FP16_REPORT.md) | `stable_run.py` | Images and saved splits |
| Bounded cubic FP16 training and stress tests | [Cubic stability](BOUNDED_CUBIC_REPORT.md) | `cubic_run.py` | Images and saved splits |
| INT8 internal forward arithmetic | [Integer arithmetic](INT8_INTERNAL_REPORT.md) | `int8_internal_run.py` | `cubic_results/bounded_{11,22,33}.pt` |
| INT8-aware training from scratch | [QAT](QAT_REPORT.md) | `qat_run.py` | Images and saved splits |

These are numerical CPU experiments, not GPU throughput measurements. FP8 storage rounding is not an FP8 GEMM. INT8 products use widened INT32 operands/accumulation. QAT gradients and master parameters remain floating point. Reports distinguish frozen-checkpoint ablations from actual training. Tiny quality changes do not establish superiority.

## Reproduce

Use a disposable copy of this directory: original runners write their fixed result paths and will replace captured JSON. Python dependencies are listed in `requirements.txt`. From the repository root, prepare the hash-verified corpus:

```bash
python experiments/denoising/quantization/prepare_data.py
cd experiments/denoising/quantization
python stable_run.py
python stable_checks.py
python cubic_run.py
python cubic_checks.py
python int8_internal_run.py
python qat_run.py
python qat_checks.py
```

For the frozen-checkpoint precision ablations, restore the original R8S32 checkpoints if available, or regenerate them with `python rank_run.py` (which trains four historical compact configurations, not the MLP/reference-KS2 controls), then run `python quant_ablation.py` and `python quant_diagnostics.py`. `quant_initial.py` is the original pre-GS128 coefficient grouping; `quant_ablation.py` contains the final GS128 policy. Existing controls are recorded under `results/`; none were rerun to prepare this PR.

Datasets, model checkpoints, compiled libraries, and machine-specific ZIP-packaging scripts are not committed. Historical reports refer to their original downloadable archives when mentioning bundled images/checkpoints or `*_report.py`; those archive statements do not describe this Git tree. Run the numerical scripts above, which directly write JSON. `source_manifest.json` records byte hashes of the captured files; model and experiment scripts retain their original bytes. The shared denoising README contains the paper citation and full evaluation protocol.
