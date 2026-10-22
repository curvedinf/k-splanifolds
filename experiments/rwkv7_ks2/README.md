# RWKV-7: KS2 memory-controller ablation

Follow-up: [KS2 controllers plus quarter-sized KS2 FFNs](FFN_REPORT.md). The
original controls below were reused without retraining.

Further follow-up: [FFN width screen and update-scale diagnostic](WIDTH_REPORT.md).

Latest follow-up: [controlled FFN update scales](CONTROLLED_WIDTH_REPORT.md).

Recorded findings: replacing just the memory controllers cuts their parameters
65% with essentially unchanged three-seed loss (2.09809 versus 2.09765). Adding
quarter-sized KS2 FFNs raises loss to 2.17355. A single-seed width screen and
width-scaled output-projection updates improve the best FFN result to 2.15819,
still above the original seed-11 loss of 2.09916. These are short-context,
1,000-step experiments; no large-model or long-context advantage is established.

## Files and setup

`run.py` supports `--variant mlp`, `ks2` (controllers), and `ks2_ffn`
(controllers plus FFNs). `--ffn-charts` sets FFN width; `--control-update-scale`
scales its output-projection learning rate inversely with width. Output directories
should be distinct for different widths/settings to preserve each checkpoint.

Install `requirements.txt` and build the existing kernels before running. The
runner requires an AVX-512 CPU for KS2 and two local FineWeb GPT-2 shard files named
`fineweb_train_000001.bin` and `fineweb_val_000000.bin`; their pinned download source
is linked below. Alternatively provide prepared `train.bytes`, `val.bytes`, and
`data_manifest.json` from a prior run. The runner records the decoded byte hashes.

```bash
python -m pip install -r experiments/rwkv7_ks2/requirements.txt
python experiments/common/ks2_cpu/build.py
python experiments/rwkv7_ks2/check.py
python experiments/rwkv7_ks2/check_optimizer.py
```

Use `diagnose_width.py --data /path/to/prepared-data` and
`diagnose_controlled_width.py --data /path/to/prepared-data` for the first-update
measurements. The JSON reports contain configurations, final measurements, and
full validation curves. Historical source-hash files describe their recording
snapshots; `controlled_source_hashes.json` records the final runner and checks.
Large checkpoints, prepared datasets, and compiled binaries are not committed.

This is a small, from-scratch CPU experiment replacing only the decay and
in-context learning-rate controllers with stabilized KS2. It is not a pretrained
RWKV checkpoint evaluation, production GPU speed comparison, or long-context test.

## Design

- Two layers, width 128, four heads of size 32, 256 byte tokens, untied output head.
- RWKV-7 generalized delta-rule recurrence, six learned token shifts, normalized
  removal keys, key modulation, first-layer value residual, output gate, bonus
  readout, group normalization, and ordinary 4x ReLU-squared channel-mixing FFN.
- Baseline: rank-32 tanh decay controller and rank-32 linear update controller,
  followed by their original bounded output transforms. Calling the baseline
  "MLP" refers to this original controller pair; the update controller has no
  intermediate activation.
- KS2: one rank-8 stabilized subspace per controller, one geometry step, fused CPU
  forward and analytic backward. Projections and geometry train in FP32.
- All shared parameters and initial controller outputs are identical for each
  paired seed. Baseline uses zero input projections and nonzero output projections
  following upstream initialization; KS2 uses nonzero input projections and zero
  output projections so its initial control contribution is also exactly zero.
  This preserves initial behavior but is not identical parameterization or gradient dynamics.
- Both retain the same decay bias and exponential-of-bounded-sigmoid transform,
  sigmoid update-rate bound, and normalized keys. The recurrent state stays FP32.
- The shared value-residual and output-gate controllers remain unchanged.

## Data and training

Use the first 1,000,000 GPT-2 tokens from `fineweb_train_000001.bin` and the first
100,000 from the separate `fineweb_val_000000.bin` shard of
[kjj0/fineweb10B-gpt2](https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/tree/889765ea1f903759787add96995d81171b632d0c).
Decode each token to its raw bytes, including special-token byte spellings. This
is byte-level language modeling, so loss cannot be compared numerically with our
earlier GPT-2-token nanoGPT results. Byte hashes and exact sizes are in results.json.

Three paired seeds: 11, 22, 33. Each model trains for 1,000 steps, batch 4,
sequence length 64 (256,000 byte targets per run), with zero state at each sampled
window. Training samples are drawn with replacement, identically within each seed.
AdamW: learning rate 0.001 decaying by cosine to 0.0002, default betas, weight decay
0.01, gradient clipping at norm 1. No learning-rate or architecture search.

Validation uses the same 32 fixed batches (8,192 byte targets) for every model.
Report the final checkpoint, not the best checkpoint. Validation windows may
overlap within the held-out shard. No final independent test set is used.

CPU: AMD EPYC 9V74, two PyTorch threads and one fused-geometry thread. Runs execute
sequentially. Training time includes forward/backward/optimizer, excludes data
preparation and validation. Inference is median latency over 20 timed forwards
after five warmups, batch 4 x 64 tokens, with a fresh zero state each forward.
This is sequence inference, not cached single-token decoding. Both use the same
TorchScript recurrent loop; timings do not predict optimized GPU performance.
Peak RSS includes the Python process, model, data, activations, and optimizer.

## Results

| Version | Total parameters | Replaced controller parameters | Final validation loss | Training seconds | Sequence inference ms | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 537,856 | 32,768 | 2.097655 ± 0.003075 | 89.10 | 13.402 | 460.2 |
| KS2 | 516,560 | 11,472 | 2.098090 ± 0.007084 | 89.49 | 13.061 | 460.3 |

Loss is cross-entropy in natural-log units per byte; ± is sample standard deviation across seeds. Other measurements are means across runs.

| Seed | MLP loss | KS2 loss | KS2 minus MLP |
|---|---:|---:|---:|
| 11 | 2.099161 | 2.094131 | -0.005031 |
| 22 | 2.094117 | 2.093871 | -0.000246 |
| 33 | 2.099686 | 2.106269 | +0.006583 |

The mean loss difference is +0.000435 nats/byte for KS2: effectively tied at this
sample size, with KS2 better in two seeds and worse in one. Controller parameters
fall 65.0%, while total model parameters fall 4.0%. Mean training time differs by
0.4% and mean sequence inference by 2.5%; noisy per-seed timings do not establish
a speed advantage. All runs completed with finite losses and gradients.

## Scope and verification

The parameter reduction applies only to the two replaced controllers, not the
whole attention block or model. This experiment does not isolate KS2 geometry
from its bottleneck parameterization: a narrower ordinary controller is a useful
future control. Short sequences and early training cannot establish long-context
retention, large-scale convergence, or the quality of pretrained replacements.

`check.py` verifies recurrence against explicit matrix transitions, float64
gradient checking, causal prefixes, equal shared parameters, equal initial
controller outputs, and equal initial model logits. Every training step rejects
nonfinite loss or gradient norm.

## Sources and reproduction

Architecture and initialization adapted from BlinkDL's Apache-2.0 RWKV-LM:

- [Official training model](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/train_temp/src/model.py), retrieved blob `b4d167fedead2655d253c55eb47b65f00e7193d2`.
- [Official NumPy recurrence](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py), retrieved blob `9537bde2fa3cf40284d7b68fd377cf4cce39f41c`.
- [RWKV-7 paper](https://arxiv.org/abs/2503.14456).

Changes from upstream: reduced dimensions, head size 32, byte tokenizer, CPU FP32
recurrence, standalone AdamW loop without logit regularizer, no Lightning/DeepSpeed,
small uniform embedding initialization and PyTorch-default output-head initialization.
All these choices are shared between the experimental arms. See LICENSE.RWKV-LM.

From the repository root, with PyTorch, NumPy, tiktoken, and a C++ compiler installed
on an AVX-512 CPU:

```bash
python experiments/common/ks2_cpu/build.py
python experiments/rwkv7_ks2/check.py
for seed in 11 22 33; do
  for variant in mlp ks2; do
    python experiments/rwkv7_ks2/run.py --data /path/to/fineweb-shards --out runs/rwkv7 --variant "$variant" --seed "$seed" --steps 1000
  done
done
python experiments/rwkv7_ks2/summarize.py runs/rwkv7
```

Checkpoints contain model state only, not optimizer state. Raw curves, manifests,
and configurations are in results.json; aggregate values are in summary.json.
The downloadable bundle includes prepared bytes under `data/`; pass that directory
to `--data` to reproduce without downloading the original shards. Geometry changes
relative to initialization are recorded in geometry_changes.json; every geometry
parameter tensor changed in all three KS2 seeds.
