# Fused KS3 against the nanoGPT MLP control

After optimizing KS3 and enabling direct GPU graph replay, training averaged **74.27 ms/step**, versus **76.64 ms/step** for the existing MLP control: **3.1% less step time**. This is a modest speed lead in this specific small-model scenario. KS3 did **not** win on loss: **7.849194 vs 7.801634** after 500 steps.

## Matched scenario

The reference is this directory’s original `config.json` and the MLP entries in `results.json`, not an RWKV or sibling nanoGPT control. Both new arms use its width-96, eleven-FFN topology, sequence length 128, seeds 11/22/33, 500 optimizer steps (64,000 training tokens), 2,048 validation tokens, full 50,257-token validation vocabulary, and original ANVIL/Adam learning-rate and objective schedules. MLP retains its 96→352→96 ReLU² FFNs and original initialization. Shared weights are initialized identically before constructing the interchangeable FFNs.

KS3 uses 337 rank-1 subspaces per FFN at the full MLP FFN budget, GS32 W8A8 projections, FP16 quantization scales and nonlinearities, and FP32 masters, backward and optimizer state. The rest of the model stays FP32.

| Model | Parameters per FFN | All 11 FFNs | Training step | Full 500-step run | Final validation CE |
|---|---:|---:|---:|---:|---:|
| Original MLP control, GPU rerun | 67,584 | 743,424 | 76.64 ms | 39.90 s | 7.801634 |
| Optimized KS3 + graph replay | 67,400 | 741,400 | 74.27 ms | 39.21 s | 7.849194 |

Loss is the mean of three unique seeds. Timings average two complete repetitions of those seeds, with GPU assignments swapped in the second repetition. The repetitions produce identical losses and are not six independent quality seeds. Four MI100s run independent complete models, one job per GPU; this is experiment parallelism, not tensor-parallel nanoGPT. The shared kernel’s TP4 path was checked separately.

Training timing synchronizes each full forward/backward/optimizer step, excluding validation. Total run time includes model construction/capture, validation and checkpoint saving; disposable pilots and data loading are outside it. The initial and subsequent comparisons each rerun the unchanged MLP to account for host/hardware timing variation.

The historical CPU MLP reference was **7.819336 CE**, **218.80 ms/training step**, and **112.86 s/run**. Its time is not used as the GPU speed baseline. GPU candidate sampling and floating-point kernels differ from the historical CPU execution, so its loss need not reproduce bit-for-bit. Dataset SHA256 hashes match the historical manifest.

## Optimization sequence

| Candidate | MLP ms/step | KS3 ms/step | KS3 / MLP time | KS3 final CE |
|---|---:|---:|---:|---:|
| initial | 72.08 | 92.23 | 1.280× | 7.840190 |
| packing_backward | 76.97 | 85.01 | 1.104× | 7.849194 |
| geometry_adam | 75.53 | 83.68 | 1.108× | 7.849194 |
| torch_compile | 72.50 | 78.53 | 1.083× | 7.876231 |
| graph | 76.08 | 74.32 | 0.977× | 7.849194 |
| graph_repeat | 77.21 | 74.22 | 0.961× | 7.849194 |

1. Replaced many small Torch weight-packing operations with batched quantization, then fused activation, both projection and geometry packing into one launch.
2. Combined the three independent backward GEMMs and geometry reduction. Rank-1 forward/backward now uses five GPU launches; other ranks retain their chart-mean calculations.
3. Fused the existing three geometry Adam updates per FFN, preserving parameters, FP32 moments, bias correction, learning rates and decay. Projection ANVIL and MLP calculations remain unchanged.
4. Tested `torch.compile(mode="reduce-overhead")`. It helped latency but produced a different loss trajectory; it is not the selected path.
5. Used direct forward/backward graph replay. Every validation checkpoint exactly matches optimized eager training for all three seeds, in both repetitions.

The optimized eager implementation has small FP32 rounding differences from the initial implementation; its loss is reported separately rather than assumed identical. In a trained-cell diagnostic, packed codes/scales and forward outputs matched exactly, while the largest relative gradient difference was about 1.8e-7. See [numeric check](numeric_comparison.txt). That local check does not establish bitwise-equivalent long training.

## Per-seed results

| Seed | MLP CE | KS3 CE | KS3 − MLP | MLP ms/step, two replays | KS3 ms/step, two replays |
|---|---:|---:|---:|---:|---:|
| 11 | 7.796580 | 7.852084 | +0.055504 | 76.84 / 77.43 | 71.41 / 75.46 |
| 22 | 7.860978 | 7.925310 | +0.064331 | 74.56 / 77.48 | 75.33 / 72.96 |
| 33 | 7.747344 | 7.770189 | +0.022845 | 76.83 / 76.72 | 76.23 / 74.23 |

## Reproduce and validate

```bash
python -m experiments.nanogpt_ks2.compare_ks3 \
  --data-dir /path/to/fineweb10B --out experiments/nanogpt_ks2/runs/ks3-control-new \
  --gpus 0 1 2 3 --ks3-graph
```

Repeat with `--gpus 1 0 3 2` and a new output directory. The comparison runner reads the saved original scenario and always uses exactly 500 steps; it does not adapt the step count to a time budget. ROCm 7.1, PyTorch 2.11.0+rocm7.1, AMD Instinct MI100.

All 18 common tests, seven nanoGPT control/integration tests, and TP4 output/input/parameter-gradient checks passed. These cover arbitrary widths/ranks, partial GS32 groups, AMP, compilation, checkpoint reload, graph replay after weight updates, other-shape fallback, optimizer ownership, causality, and geometry optimizer parity.

Graph capture is optional and requires one use of each captured FFN before its backward, as in this nanoGPT model. Evaluation, autocast and different input shapes use the normal fused path. Ordinary fused KS3 supports multiple outstanding forwards. Checkpoints retain ordinary parameter names and contain no captured graphs.

This remains a 64,000-token, short-context comparison on a 2,048-token validation prefix, not a convergence result or a large-model speed claim. The measured speed lead is small and varies by seed/GPU assignment.

[Summary](summary.json) · [All raw curves and timings](all_results.json) · [All candidate summaries](optimization_history.json) · [Configurations, data and source hashes](configs.json) · [Verification](verification.json) · [Isolated FFN benchmark](kernel_timing.json)
