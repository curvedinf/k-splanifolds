# RWKV-7: KS2 controllers plus KS2 FFNs

Follow-up to [the controller experiment](README.md). Only the new combined arm
was trained; the original and controller-only results below are reused unchanged.

The two ReLU-squared FFNs are replaced with stabilized KS2 cells, each with 11
rank-8 subspaces and one geometry step. Each FFN has 31,548 parameters versus
131,072 originally (24.07% of the original). Learned FFN token shifting, layer
normalization, and residual connections remain. The old FFN linear layers are
removed rather than retained as unused parameters. The decay and update-rate
controllers remain the same one-subspace KS2 cells from the preceding experiment.

FFN output projections start at zero, matching the original FFNs' zero residual
outputs. All retained parameters, including both KS2 controllers, have exactly
the same initialization as the controller-only arm. Independent FFN initialization
uses seed + 2000. This ensures equal initial model logits without claiming
identical gradient dynamics across different parameterizations.

Protocol is unchanged: FP32 CPU, fused KS2 forward and analytic backward, two
layers, width 128, head size 32, byte-level FineWeb, batch 4, sequence 64, 1,000
steps, seeds 11/22/33, identical batches, AdamW and learning-rate schedule, and
the same 8,192 held-out byte targets. Final checkpoint selection is fixed.
All data manifests and initial validation losses are checked against the old runs.

## Results

Loss is mean cross-entropy in nats/byte, with sample standard deviation across
three seeds. Timing/RSS entries are means across seeds. Inference measures a
batch of four 64-byte sequences with fresh zero states, not cached decoding.

| Version | Total parameters | FFN parameters | Validation loss | Training seconds | Inference ms | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| Original | 537,856 | 262,144 | 2.097655 ± 0.003075 | 89.10 | 13.402 | 460.2 |
| KS2 controllers | 516,560 | 262,144 | 2.098090 ± 0.007084 | 89.49 | 13.061 | 460.3 |
| KS2 controllers + FFNs | 317,512 | 63,096 | 2.173550 ± 0.002201 | 91.16 | 14.661 | 460.3 |

| Seed | Original loss | KS2 controllers | KS2 controllers + FFNs | Combined minus controllers |
|---|---:|---:|---:|---:|
| 11 | 2.099161 | 2.094131 | 2.171209 | +0.077079 |
| 22 | 2.094117 | 2.093871 | 2.173861 | +0.079989 |
| 33 | 2.099686 | 2.106269 | 2.175579 | +0.069310 |

Combined minus controller-only mean loss: **+0.075459 nats/byte**. Combined minus original mean loss: **+0.075895 nats/byte**. Total parameters decrease **40.97%** versus original.

These are early-training, short-context results, not a large-model or long-context
claim. No learning-rate tuning was performed for the smaller FFNs. There is no
FFN-only KS2 arm, so this tests adding KS2 FFNs on top of KS2 controllers; it does
not separate their interaction. A parameter-matched narrow ordinary FFN has not
been tested. CPU timings include the same recurrent loop and do not predict GPU
performance. Peak RSS includes data, activations, optimizer state and runtime;
it is not just weight storage. Prior timings were recorded in earlier processes.

## Reproduce only the new arm

From the extracted bundle root (or repository root with the corresponding files):

```bash
python experiments/common/ks2_cpu/build.py
python experiments/rwkv7_ks2/check.py
for seed in 11 22 33; do
  python experiments/rwkv7_ks2/run.py --data data --out runs/rwkv7-ffn --variant ks2_ffn --ffn-charts 11 --seed "$seed" --steps 1000
done
python experiments/rwkv7_ks2/summarize_ffn.py runs/rwkv7-ffn
```

The prepared byte files in the bundle match the original manifests exactly.
For the same peak-process-memory accounting as the recorded runs, supply the
original FineWeb shards instead of prepared bytes (the original route also loads
the tokenizer). Model/activation precision and training computation are identical.
See the original README for source attribution and common setup. New raw curves
and per-run measurements are in ffn_results.json, aggregate values in
ffn_summary.json, and source hashes in ffn_source_hashes.json.
