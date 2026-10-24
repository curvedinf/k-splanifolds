# RWKV-7: KS2 memory-controller ablation

**Current baseline:** the four-layer demuxed-history model with 96D main,
2048D decoder-only side residual, 64 rank-8 KS2 subspaces per decoder,
and learned dense RWKV projections (567,552 parameters). See
[current_baseline.json](current_baseline.json) for the constructor and protocol.

Follow-up: [KS2 controllers plus quarter-sized KS2 FFNs](FFN_REPORT.md). The
original controls below were reused without retraining.

Further follow-up: [FFN width screen and update-scale diagnostic](WIDTH_REPORT.md).

Follow-up: [controlled FFN update scales](CONTROLLED_WIDTH_REPORT.md).

Follow-up: [15,000-step training on four MI100 GPUs](GPU_TRAINING_REPORT.md).
Three paired seeds and GPU reruns of the 1,000-step controls show that longer
training narrows but does not eliminate the FFN gap: final mean losses are
1.66140 (original), 1.66417 (KS2 controllers), 1.72761 (11-subspace KS2 FFNs),
and 1.70957 (22-subspace KS2 FFNs with scaled output updates).

Follow-up: [four width-96 layers with KS2 only in the middle FFNs](HYBRID4_REPORT.md).
At 15,000 steps over three paired seeds, the all-MLP control reaches 1.65210
and the MLP/KS2/KS2/MLP hybrid reaches 1.66383. Each KS2 FFN uses 51.13% of
the replaced MLP's parameters; total parameters decrease 11.70%.

Follow-up: [middle KS2 FFN budgets from full to one sixteenth](HYBRID4_BUDGET_REPORT.md).
At 15,000 steps over the same three paired seeds, full/half/quarter/eighth/sixteenth
KS2 budgets give mean losses 1.66111/1.66383/1.66720/1.67195/1.67373,
versus 1.65210 for all MLP. The two smallest budgets are close and reverse order
on the larger validation sample. Budget fractions apply only to the middle FFNs.

Follow-up: [105,000-step continuation results](hybrid4_plateau_summary.json)
and [loss curves](hybrid4_plateau_curves.png). Each configuration resumes its
three paired 15k checkpoints (seeds 11, 22, 33), including Adam state and the
original minibatch stream, and trains to exactly 105,000 total steps. Base LR
stays at the previous cosine schedule's terminal 0.0002; the existing KS2
output-projection LR and weight-decay scaling is preserved. All runs retain
four width-96 layers, original memory controllers, full outer MLP FFNs, and
learned byte embeddings and logits heads.

The following losses use the same larger, fixed 65,536-byte validation sample
at both endpoints; ± is sample SD across three seeds. These differ from the
8,192-target validation figures in the preceding summaries.

| Middle FFNs | Total parameters | Validation at 15k | Validation at 105k | Sampled training at 105k |
|---|---:|---:|---:|---:|
| Full MLP | 615,936 | 1.638507 | 1.560535 ± 0.002844 | 1.406925 |
| Half-budget KS2 | 543,872 | 1.654041 | 1.572509 ± 0.000844 | 1.425591 |
| Sixteenth-budget KS2 | 477,904 | 1.661134 | 1.579496 ± 0.004329 | 1.436874 |

Longer training narrows the half-budget gap from 0.015534 to 0.011974 nats/byte
and the sixteenth-budget gap from 0.022627 to 0.018960. MLP remains better in
every paired seed on both validation samples. The original 8,192-target final
means are 1.566504 / 1.578159 / 1.588292 for MLP / half / sixteenth.

The user selected the existing 105k checkpoints as the stopping point. The
sixteenth model had not met the planned validation plateau criterion: its best
three-seed mean still improved by 0.006642 over the last 20k steps, above the
0.002 threshold. Its sampled training loss also continued falling. This is a
matched-step comparison at the requested cap, not evidence of full convergence.
The validation sample comes from the same existing held-out shard, not an
independent test set.

All nine final checkpoints have Adam counters at 105,000, finite parameters,
matching model/data sources, and matched sampler states within each seed.
Reloaded checkpoints reproduce all three logged evaluation metrics. Resume
checks compared captured and eager updates, including Adam moments. MLP seed 11
was additionally resumed at 100k for its final 5k steps. Checkpoints and logs
are under ignored `gpu_runs/hybrid4_plateau/`; portable archives are
[raw histories](hybrid4_plateau_results.json), [statistics](hybrid4_plateau_summary.json),
and [checkpoint reevaluation](hybrid4_plateau_evaluation.json).

To reproduce from the existing 15k checkpoints into a fresh directory:

```bash
../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/sweep_plateau.py --out experiments/rwkv7_ks2/gpu_runs/hybrid4_plateau_repro --max-steps 105000
../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/summarize_plateau.py --runs experiments/rwkv7_ks2/gpu_runs/hybrid4_plateau_repro
```

Latest follow-up: [full MLP plus a sixteenth-budget KS2 branch](additive4_summary.json)
and [comparison curves](additive4_curves.png). The middle two layers now use
`MLP(h) + KS2(h)` on the same normalized, time-mixed input. Both full MLPs are
retained; outer layers remain MLP. The extra KS2 branches start at zero output,
preserving all original shared weights and initial logits. They add 9,424
parameters, raising the total from 615,936 to 625,360 (+1.53%). The same three
seeds, data stream, 15k cosine schedule, and 90k constant-LR continuation are
used. Only the added KS2 output projections receive the prior 5.5x LR multiplier
and inverse weight-decay scaling; the original MLP learning rates are unchanged.

Final validation cross-entropy at exactly 105,000 steps, averaged over seeds
11/22/33; ± is sample SD across seeds:

| Model | Fixed 65,536-byte sample | Nearly complete validation shard, four offsets |
|---|---:|---:|
| All MLP | 1.560535 ± 0.002844 | 1.540879 ± 0.003010 |
| Full MLP + 1/16 KS2 in middle FFNs | 1.557408 ± 0.003354 | 1.540212 ± 0.003290 |

The additive variant is nominally better by 0.003128 on the 65k sample, but worse
by 0.002951 on the original 8,192-target sample (1.569455 versus 1.566504).
To investigate this disagreement, the existing 444,316-byte validation shard was
tiled with 64-target windows at offsets 0, 16, 32, and 48. This covers nearly all
bytes four times with different contexts (1,777,024 target visits per model),
not independent samples or a new test set. The additive mean advantage there is
only 0.000668, with wins in two of three seeds. Batch-64 evaluation agreed with
batch-4 per-window losses to within 2.4e-7 in the parity check.

A fourth GPU ran an extra MLP seed-11 repeat. GPU training is not bit-repeatable
from scratch: even two runs of the archived, unchanged model diverged in a short
[repeatability diagnostic](mlp_reproducibility.json). Archived/current MLP initial
weights, logits, and gradients were exactly equal. The full-length MLP repeat
shifted the four-offset validation loss by -0.001841 relative to its archived
seed-11 result, larger than the additive variant's mean advantage. The repeat is
kept separate from the three-seed averages. These results do not establish a
reliable quality improvement from the extra branch; a parameter-matched wider
MLP was not tested.

The added KS2 branch is used: zeroing its output projections **only at evaluation**
raises the 65k-sample mean from 1.557408 to 1.567699 (+0.010291). This measures
dependence of the co-adapted model on its branch, not the effect of retraining
without KS2. Weights were restored and original losses rechecked afterward.

All three additive checkpoints and the extra MLP checkpoint reached 105k with
finite parameters, matching sampler states and data, and verified optimizer
counters. Reloaded checkpoints reproduce all logged evaluation metrics. The MLP
repeat resumed from its saved 30k state after a stalled process; the additive
runs were uninterrupted. Prior model/runner sources are preserved under
`source_snapshots/pre_additive/`. Artifacts: [raw histories](additive4_results.json),
[checkpoint/branch evaluation](additive4_evaluation.json), and
[four-offset validation](additive4_tiled_validation.json).

Training entrypoint (use fresh directories):

```bash
../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/sweep_additive.py --data experiments/rwkv7_ks2/gpu_runs/data --initial-out experiments/rwkv7_ks2/gpu_runs/additive4_initial_repro --out experiments/rwkv7_ks2/gpu_runs/additive4_repro
```

Original CPU findings: replacing just the memory controllers cuts their parameters
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

## Hadamard multiplexing into two rank-8 subspaces

The [projection experiment](spread2_summary.json) keeps the four-layer,
width-96 model and replaces projections only in the middle two KS2 FFNs.
Each still has **two rank-8 subspaces**, with every one of the 96 input
features contributing to every coordinate in both subspaces.

The encoder adapts `SpreadViews` from
[`RHAC/src/rhac/rhac.py`](../../../SDGraft/RHAC/src/rhac/rhac.py), not the
balanced-bucket implementation in `postdown.py`. Each subspace has an
independent fixed permutation and sign pattern. Apply normalized eight-way
Hadamard mixing to twelve groups, apply the signs, then sum the twelve
groups and divide by `sqrt(12)`. Every resulting coefficient is
`+/-1/sqrt(96)`, and the eight rows of each subspace code are orthonormal.
The view seed is `4242 + 99173`, shared across training seeds and layers.
This tests the encoder on current FFN inputs; it adds no depth-history bank.

The fixed-output variant uses the transpose code with unit row norms,
followed by 96 learned scalar output gains initialized to zero. The input
and output mixing matrices are fixed. All variants have identical initial
logits; shared weights and KS2 geometry initialization are paired.

Twelve fresh runs used three seeds (11/22/33), 15,000 steps and 3.84 million
training targets each. Four independent runs occupied all four MI100 GPUs;
MLP took 110–116 seconds per run and KS2 130–139 seconds. The existing
AdamW/cosine protocol was retained, including the 5.5x output-weight/gain
learning rate and inverse weight-decay adjustment. The code changes input
scale and output parameterization; these were not separately tuned.

| Variant | Total trainable parameters | 65k validation | Full-shard validation, mean ± seed SD |
|---|---:|---:|---:|
| All MLP | 615,936 | 1.640852 | 1.622722 ± 0.001068 |
| KS2, learned projections | 477,904 | 1.664924 | 1.645133 ± 0.000835 |
| KS2, Hadamard input / learned output | 474,832 | 1.666996 | 1.649524 ± 0.002225 |
| KS2, Hadamard input / output + scalar gains | 471,952 | 1.664794 | 1.646821 ± 0.001681 |

Fixed input/output nearly matches learned projections, but does not close
the MLP gap in this short test. Its full-shard paired difference from
learned KS2 is +0.001688 nats/byte; fixing only the input costs +0.004391.
The tiny improvement on the 65k sample does not persist on the full shard.
Full-shard evaluation uses offsets 0/16/32/48 and 1,777,024 target visits
per model; these overlap and are not independent test examples.

Checks passed for RHAC reference forward/input-gradient equivalence,
all-input coverage, per-chart orthogonality, paired initialization, initial
logits, optimizer coverage and nonzero geometry gradients. All 12 runs
completed with finite parameters/gradients and exact 15k optimizer counters.
Checkpoint reevaluation reproduced training-run validation metrics. Batch-64
full-shard losses matched batch-4 losses within 4.77e-7 on the checked batch.

[Implementation](spread_projection.py) · [Checks](check_spread.py) ·
[Raw runs](spread2_results.json) · [Full-shard evaluation](spread2_tiled_validation.json) ·
[Curves](spread2_curves.png)

```bash
../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/check_spread.py
../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/sweep_spread.py --out experiments/rwkv7_ks2/gpu_runs/spread2_repro --steps 15000
../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/evaluate_spread.py --runs experiments/rwkv7_ks2/gpu_runs/spread2_repro
../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/summarize_spread.py --runs experiments/rwkv7_ks2/gpu_runs/spread2_repro
```

## Hadamard multiplexing with 32 rank-8 subspaces

The [32-subspace follow-up](spread32_summary.json) retains width 96 and the
same 15,000-step protocol, while providing 32 independent fixed codes of
all 96 features. Each inner FFN has 26,240 geometry parameters and 96
zero-initialized output gains: **26,336 trainable parameters per inner FFN,
521,152 for the model**. The historical half-budget learned-projection
variant had 16 rank-8 subspaces and 37,696 parameters per inner FFN.

All 32 codes are distinct, have orthonormal rows within each subspace, and
include every input in every coordinate. Fixed decoder rows retain unit
norm, so output gains keep the two-subspace experiment's 5.5x LR and inverse
weight decay; they do not acquire the dense output matrix's `11/S` scaling.
The extra learned-projection 32-chart control uses `11/32` output LR scaling.

| Variant | Seeds | Total parameters | 65k validation | Full-shard validation |
|---|---:|---:|---:|---:|
| Hadamard input/output, 2 charts | 3 | 471,952 | 1.664794 | 1.646821 ± 0.001681 |
| Hadamard input/output, 32 charts | 3 | 521,152 | 1.662357 | 1.644048 ± 0.004457 |
| Learned input/output, 32 charts | 1 (11) | 619,264 | 1.647429 | 1.628228 |
| All MLP, reused from the preceding short study | 3 | 615,936 | 1.640852 | 1.622722 ± 0.001068 |

The mean improvement from 2 to 32 fixed charts is modest relative to seed
variation. It does not close the MLP gap. The same-geometry learned control
is better for seed 11, but has only one seed and 98,112 more parameters.
All four new runs completed with finite losses/gradients, verified saved
15k-step optimizer counters, matching checkpoint reevaluation, and full-shard
evaluation. Training took 144–150 seconds per run across four MI100s.

[Raw results and per-seed full-shard evaluations](spread32_results.json) ·
[Runner](sweep_spread32.py). The experiment used:

```bash
../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/sweep_spread32.py --out experiments/rwkv7_ks2/gpu_runs/spread32_short --steps 15000
```

Use a fresh directory for another run. Per-run full-shard evaluation is
available through `evaluate_spread.py --runs <directory> --arm fixed_both
--seeds 11 22 33` (and `--arm learned --seeds 11` for the control).

## Four KS2 FFNs with an FFN-only side residual

[`SideModel`](side_stream.py) fixes the main residual stream, embedding,
logits head, and all four original RWKV time-mixing modules at width 96.
All four FFNs are KS2 with 32 rank-8 subspaces. An additional side stream
is initialized to zero and carried between FFNs, without entering any
attention module or the embedding/logits head directly:

```text
main = main + RWKV(LayerNorm96(main))
joined = concat(main, side)
joined = joined + KS2(FFN_time_mix(LayerNorm(joined)))
main, side = split(joined, [96, side_width])
```

The final head reads only `main`; final side outputs are discarded.
Gradients are retained across every side connection. FFN normalization
and causal time mixing operate over the combined stream. Main time-mix
parameter initialization is preserved across side widths. Shared attention,
embedding/head, and all KS2 geometry tensors have exactly paired initial
values. Fixed projection codes independently permute, Hadamard-mix, sign,
and fold every combined input feature into every subspace coordinate.

| Additional side channels | FFN interface | Trainable parameters |
|---:|---:|---:|
| 0 | 96 | 426,368 |
| 32 | 128 | 426,880 |
| 96 | 192 | 427,904 |
| 160 | 256 | 428,928 |
| 2,048 | 2,144 | 459,136 |

The parameter formula is `426368 + 16 * side_width`. All versions retain
269,568 RWKV parameters, 49,152 embedding/head parameters and 104,960 KS2
geometry parameters. The added parameters are FFN normalization, time-mix
vectors and output gains. Counts include the final FFN's side gain slots,
which have zero task gradient because their outputs are discarded.

The short protocol uses 15,000 steps and the existing FP32 AdamW schedule,
with the fixed output gains retaining their 5.5x LR and inverse weight
decay adjustment. The 0/32/96/160 sweep uses seeds 11/22/33, one concurrent
configuration per MI100. The zero-side model is the architecture-matched
control: the older tests retained full MLPs in the outer two layers.

[`check_side.py`](check_side.py) verifies fixed attention/head dimensions,
four KS2 FFNs, distinct codes and all-input coverage, paired initial
parameters/logits, side-to-FFN gradient flow, no direct side-to-attention/head
path, discarded final side outputs, and full-model GPU forward/gradient
agreement with the reference recurrence.

```bash
../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/sweep_side.py --out experiments/rwkv7_ks2/gpu_runs/side_stream_short
../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/summarize_side.py --runs experiments/rwkv7_ks2/gpu_runs/side_stream_short
```

Use a fresh output directory to reproduce. Per-run checkpoints, expanded
train/validation evaluations and four-offset full-shard evaluations are
stored together beneath each side-width directory. Full-shard offsets are
0/16/32/48; overlapping target visits are not independent test examples.

Completed 15,000-step results (three seeds per row):

| Side channels | Parameters | 65k validation | Full-shard validation, mean ± seed SD | Mean sustained ms/step |
|---:|---:|---:|---:|---:|
| 0 | 426,368 | 1.704475 | 1.684595 ± 0.001342 | 12.116 |
| 32 | 426,880 | 1.699525 | 1.682143 ± 0.000971 | 11.889 |
| 96 | 427,904 | 1.699640 | 1.681925 ± 0.001134 | 12.307 |
| 160 | 428,928 | 1.698953 | 1.682453 ± 0.001981 | 12.353 |

Every side-stream arm improves on the no-side control in all three paired
seeds, by 0.0021–0.0027 nats/byte on average. Increasing side width beyond
32 does not give a clear monotonic improvement in this short test. The
fixed code changes with interface width, and only one coding seed was
tested. Timing above is from simultaneous jobs on different MI100s; the
small apparent speed advantage of side 32 should not be treated as a
demonstrated architecture speedup.

All twelve checkpoints reached 15k with finite losses/gradients and verified
optimizer counters. Checkpoint reevaluation reproduced logged validation;
full-shard batch-64 evaluation was checked against batch 4. Shared initial
validation loss is exactly equal across side widths within every seed.

[Summary](side_stream_summary.json) · [Raw results](side_stream_results.json) ·
[Curves](side_stream_curves.png)

After all twelve runs finished, a single seed-11 run with 2,048 additional
side channels (2,144 total FFN input/output channels, main stream still 96)
completed 15,000 steps. It has 459,136 trainable parameters. Sequential
fresh-process benchmarks on the same physical MI100 GPU0 measured:

| Side channels | FFN interface | Benchmark ms/step |
|---:|---:|---:|
| 0 | 96 | 12.060 |
| 160 | 256 | 12.234 |
| 2,048 | 2,144 | 13.654 |

The 2,048-side model is 13.2% slower than no side and 11.6% slower than
side 160 in these 100-update benchmarks. Its full training run averaged
14.055 ms/step (210.825 seconds of training). Measurements include batch
copies, LR writes, forward, loss, backward, clipping and AdamW; setup and
evaluation are excluded. Benchmark updates are reset before training.

Its expanded 65k validation loss is 1.698509 and four-offset full-shard
loss is 1.678520. The paired seed-11 full-shard controls are 1.685905
(no side) and 1.684622 (side 160). This is one seed, so the improvement
needs replication before drawing a conclusion about the larger width.
Checkpoint reevaluation matches the final logged validation loss; full-shard
batch-64 versus batch-4 loss differs by at most 7.2e-7.

```bash
../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/run_side_large.py --sweep experiments/rwkv7_ks2/gpu_runs/side_stream_short --out experiments/rwkv7_ks2/gpu_runs/side_stream_2048
```

[2,048-side summary](side_stream_2048_summary.json) ·
[2,048-side raw results](side_stream_2048_results.json)

### Doubling subspaces with 2,048 side channels

With main width 96 and side width 2,048 fixed, doubling every KS2 FFN
from 32 to 64 rank-8 subspaces improves full-shard validation loss by
0.008424 in the matched seed-11, 15,000-step run:

| Subspaces per FFN | Parameters | Full-shard validation | 65k validation | Benchmark ms/step | Sustained ms/step |
|---:|---:|---:|---:|---:|---:|
| 32 | 459,136 | 1.678520 | 1.698509 | 13.654 | 14.055 |
| 64 | 564,096 | 1.670096 | 1.686523 | 14.753 | 15.419 |

Parameters increase 22.9%; benchmark step time increases 8.0% and sustained
step time increases 9.7%. Both runs used physical GPU0, the same data,
seed, training schedule, FP32 precision and fixed-output-gain LR multiplier
of 5.5. Each of the 64 subspaces still receives all 2,144 FFN input features.
RWKV time-mixing modules and embedding/logits remain at width 96.
There is one training seed per configuration. The code generator's draw
ordering means the fixed codes change when subspace count changes;
geometry initialization also changes shape. This comparison tests the
whole 64-subspace configuration, not merely appending 32 identical views
to an otherwise identical initialized model.

Checks verified the original 32-subspace state dict remains exactly
unchanged under the generalized constructor, all-input coverage and
orthonormal rows of all 64 codes, unit-norm output rows, identical initial
logits, and finite geometry/side gradients. The completed run had zero
invalid steps; saved optimizer counters, source hashes and checkpoint
reevaluation were verified. Full-shard batch-64 versus batch-4 loss differed
by at most 7.2e-7. Prior 32-only source files are preserved under
`source_snapshots/side_stream_32charts/`.

```bash
../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/run_side_subspaces.py --out experiments/rwkv7_ks2/gpu_runs/side_stream_2048_c64
```

[64-subspace summary](side_stream_2048_c64_summary.json) ·
[64-subspace raw results](side_stream_2048_c64_results.json)

### Separate attention and FFN history inputs

[`HistoryModel`](side_history.py) starts from the 64-subspace, rank-8,
2,048-side configuration. For every strictly previous layer it appends two
96D main-stream states to the current FFN input: the residual immediately
after the RWKV addition, and the residual immediately after the KS2 FFN
addition. These are separate entries, ordered by layer and then sublayer.
They are post-addition residual states, not isolated branch updates.

```text
history = []
for layer in layers:
    attention_state = main + RWKV(norm(main))
    current = concat(attention_state, side)
    input = concat(current, *history)
    current += KS2(norm_and_time_mix(input))
    main, side = split(current, [96, 2048])
    history.extend([attention_state, main])
logits = head(norm(main))
```

Hadamard input widths are 2,144 / 2,336 / 2,528 / 2,720; every input enters
every coordinate of all 64 rank-8 subspaces. All output widths remain
2,144. The baseline output matrices, gains, KS2 geometry initialization,
attention, embedding/head, and current-stream time-mix initialization are
preserved. New input codes use the existing width-dependent generator;
the widened input codes are not paired copies of the old codes.
LayerNorm and causal FFN time mixing operate on the entire expanded input.
History retains gradients and is read-only; the model never writes FFN
outputs into past history entries. Attention directly sees only the current
96D main stream. Final side outputs remain discarded.

There are 567,552 trainable parameters, 3,456 (+0.61%) above the no-history
baseline, from added normalization and time-mix parameters. This experiment
tests history together with widened normalization/mixing and changed input
codes; it does not isolate each of those effects independently.

[`check_side_history.py`](check_side_history.py) checks exact history contents
and ordering, all-input code coverage and row orthogonality, paired baseline
parameters/initial logits, nonzero gradients into all six history slices
read by the final FFN, causal forward behavior, and full-model GPU
forward/all-gradient agreement with the reference RWKV recurrence.

```bash
../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/run_side_history.py --out experiments/rwkv7_ks2/gpu_runs/side_history_2048_c64
```

Completed seed-11 comparison at 15,000 steps on the same physical GPU0:

| Configuration | Parameters | Full-shard validation | 65k validation | Benchmark ms/step | Sustained ms/step |
|---|---:|---:|---:|---:|---:|
| 64 subspaces, 2,048 side, no history | 564,096 | 1.670096 | 1.686523 | 14.753 | 15.419 |
| Separate attention + FFN history | 567,552 | 1.669112 | 1.687242 | 14.874 | 15.555 |

Full-shard loss improves by 0.000984, while the smaller 65k sample worsens
by 0.000719. This single-seed result is effectively close to the baseline;
it does not establish a reliable history benefit. Benchmark step time rises
0.82%, sustained time 0.88%. All 15k steps had finite loss/gradient norms;
checkpoint reevaluation, optimizer counters, paired initial validation,
data manifests and source hashes passed checks. Full-shard batch-64 versus
batch-4 loss differed by at most 6.0e-7.

[History summary](side_history_summary.json) · [History raw results](side_history_results.json)

### Raw KS2 history with mux before RWKV and final-only demux

[`RawMuxModel`](raw_mux.py) removes all intermediate demux projections.
Each KS2 emits its 64 × 8 = 512 coordinates directly, including the existing
coordinate identity plus geometry update. The depth history begins with
the normalized 96D embedding and accumulates each layer's separate 96D
post-attention state and 512D raw KS2 output. No history is detached.
The previous 2,048D demuxed side stream is replaced by these raw coordinates.

```text
history = normalized_embedding
for layer in layers:
    view = Hadamard_mux_96(history)
    attention_state = view + RWKV(norm(view))
    ffn_input = norm_and_time_mix(concat(history, attention_state))
    coords = Hadamard_mux_512(ffn_input).reshape(..., 64, 8)
    raw = flatten(coords + KS2_geometry(coords))
    history = concat(history, attention_state, raw)
logits = head(norm(final_demux_96(raw)))
```

The RWKV mux uses twelve independent rank-8 Hadamard views, concatenated
to 96D. Its input widths are 96 / 704 / 1,312 / 1,920. The KS2 mux uses
64 rank-8 views over inputs of width 192 / 800 / 1,408 / 2,016. All features
enter every coordinate of each mux. The final demux maps 512 coordinates
to 96D using the established fixed transpose-code matrix with unit row
norms. It has 96 learned gains initialized to one: zero initialization
would eliminate the input signal because there is no head bypass.
Intermediate raw outputs have no demux gains. The final gain LR retains
the 5.5x multiplier and inverse weight-decay adjustment.

There are 543,136 parameters: 269,568 RWKV, 209,920 KS2 geometry,
49,152 embedding/head, 96 final-demux gains, and 14,400 normalization/time
mixing. Attention and KS2 geometry initializations match the prior
64-subspace models, but initial logits differ because the residual routing
and final demux initialization change. This is a comparison of architectures,
not an isolated change with identical initial functions.

[`check_raw_mux.py`](check_raw_mux.py) verifies exact history concatenation,
pre-RWKV mux inputs, final-only demux and absence of a head bypass, all-input
Hadamard coverage/orthogonality, paired attention/geometry initialization,
gradients into every history slice and all geometries, causal forward
behavior, and GPU recurrence forward/all-gradient parity.

```bash
../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/run_raw_mux.py --out experiments/rwkv7_ks2/gpu_runs/raw_mux_64
```

Completed 15,000-step seed-11 results:

| Architecture | Parameters | Full-shard validation | 65k validation | Benchmark ms/step | Sustained ms/step |
|---|---:|---:|---:|---:|---:|
| 64-subspace/2,048-side, no history | 564,096 | 1.670096 | 1.686523 | 14.753 | 15.419 |
| Separate demuxed attention/FFN history | 567,552 | 1.669112 | 1.687242 | 14.874 | 15.555 |
| Raw KS2 history, pre-RWKV mux, final-only demux | 543,136 | 1.677181 | 1.694025 | 13.584 | 14.159 |

The raw-history architecture is faster but has 0.008070 higher full-shard
loss than the preceding history architecture in this seed. Sustained step
time is 9.0% lower and parameter count is 4.3% lower. All 15k steps had
finite loss/gradient norms. Source/data checks, checkpoint reevaluation,
and saved optimizer counters passed. Full-shard batch-64 versus batch-4
loss differed by at most 7.2e-7. One seed does not establish robustness;
several routing/initialization changes distinguish these architectures.

[Raw-coordinate summary](raw_mux_summary.json) · [Raw-coordinate results](raw_mux_results.json)

### Raw-coordinate history with a direct attention-state skip

[`RawSkipModel`](raw_mux_skip.py) makes one routing change to `RawMuxModel`:
in layers 2–4, add the immediately previous layer's 96D post-attention
state directly to the current Hadamard view, before normalization and RWKV.
The full sum also participates in the attention residual addition:

```text
view = Hadamard_mux_96(history)
combined = view + previous_attention_state  # layer 1 uses view alone
attention_state = combined + RWKV(norm(combined))
raw = KS2(norm_and_time_mix(concat(history, attention_state)))
history = concat(history, attention_state, raw)
```

The previous attention state still appears in the full history read by the
muxes. The direct skip has coefficient one and retains gradients. Raw KS2
outputs, all mux matrices, final-only demux, optimization schedule and
parameter count (543,136) are unchanged. Every initial parameter and buffer
is identical to the no-skip raw-mux baseline; initial logits differ due to
the new path. The first attention block has no skip from the embedding.

[`check_raw_mux_skip.py`](check_raw_mux_skip.py) verifies parameter/buffer
pairing, exact baseline logits with the skip disabled, correct addition
before both normalization and the attention residual, a coefficient-one
identity gradient when the mux contribution is suppressed, causality,
and GPU recurrence forward/all-gradient agreement.

```bash
../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/run_raw_mux_skip.py --out experiments/rwkv7_ks2/gpu_runs/raw_mux_skip_64
```

Completed matched seed-11 results at 15,000 steps:

| Raw-history configuration | Parameters | Full-shard validation | 65k validation | Benchmark ms/step | Sustained ms/step |
|---|---:|---:|---:|---:|---:|
| No attention skip | 543,136 | 1.677181 | 1.694025 | 13.584 | 14.159 |
| Previous attention state + current Hadamard view | 543,136 | 1.677984 | 1.697164 | 13.648 | 14.234 |

The direct skip changes full-shard loss by +0.000803 in this seed, effectively
a close result rather than a demonstrated improvement. It does not recover
the preceding demuxed-history model's 1.669112 loss. This weakens the claim
that a missing direct attention-state connection alone explains that gap;
it does not rule out a different balance between the skip and mux signals.
Benchmark and sustained step times increase about 0.5%. All 15k steps had
finite loss/gradient norms, with matching checkpoint reevaluation and
verified optimizer counters, source hashes and data manifests. Full-shard
batch-64 versus batch-4 loss differed by at most 4.8e-7.

[Attention-skip summary](raw_mux_skip_summary.json) · [Attention-skip results](raw_mux_skip_results.json)

### Independent tests of the raw-mux quality gap

[`mux_ablations.py`](mux_ablations.py) tests each proposed explanation
independently from the raw-mux baseline. Every arm has the attention-state
pass-through disabled; prior skip code and results remain preserved.
All runs use 64 rank-8 KS2 subspaces, four layers, 15,000 steps, FP32 and
the existing schedule/data/evaluation protocol.

| Arm | Only architectural change | Parameters |
|---|---|---:|
| Baseline | Original raw-mux model, no attention skip | 543,136 |
| Head residual | Add final 96D attention state to final demux before head normalization | 543,136 |
| Output gains | Multiply each raw 512D KS2 output by learned coordinate gains, initialized at 0.05 | 545,184 |
| Persistent KS2 state | Accumulate raw updates in a 512D state; store successive states in history and read the final state at the head | 543,136 |
| Source normalization | Nonaffine LayerNorm separately for each embedding, attention and KS2 source before mixing | 543,136 |

The gains use the existing 5.5x LR and inverse weight-decay rule. They start
nonzero so the gain experiment can remain independent of a head bypass.
The persistent-state arm retains the same history widths; its stored
512D entries are cumulative state snapshots instead of individual updates.
The normalization arm normalizes sources before the attention mux and,
including the current attention state, before the existing global FFN
normalization/time-mixing/mux. Stored history and final readout stay raw.
This balances per-coordinate scales, not total energy of differently sized
sources. None of these four modifications is combined with another.

The four architectural arms use training seed 11 and coding seed 103415.
To investigate seed sensitivity, both the unchanged raw-mux baseline and
the previous demuxed-history model are also run at training seeds 22/33
with coding seed 103415, and coding seeds 103416/103417 with training seed
11. Previously completed seed-11/code-103415 runs provide the references.
These are separate training-seed and coding-seed checks, not a full
factorial; their shared default point must not be counted twice as
independent evidence. Coding changes redraw all applicable fixed input and
output matrices, including the final raw-model demux. The 12 new runs are
dynamically queued with one job per GPU across all four MI100s.

[`check_mux_ablations.py`](check_mux_ablations.py) verifies exact reproduction
of both original reference constructors, independently changed routes,
absence of attention skips, paired starting parameters, finite geometry
and gain gradients, causal outputs, and GPU forward/all-gradient parity
for all four new arms.

A diagnostic of the trained no-skip raw-mux model is saved in
[`mux_source_probe.json`](mux_source_probe.json). It uses four held-out
64-token windows sampled with seed 54321. At the final RWKV input,
embedding/attention/KS2 groups contribute approximately 3.8%/24.9%/71.3%
of the sum of their separately measured projected energies. Cross-group
covariance is excluded, so these are not explained-variance fractions.
Per-coordinate RMS is about 1.17–1.41 for attention states and 0.87–1.11
for KS2 outputs, showing that equal-variance dimensional estimates are
insufficient to characterize this trained model.

```bash
../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/sweep_mux_ablations.py --out experiments/rwkv7_ks2/gpu_runs/mux_hypotheses
../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/summarize_mux_ablations.py --runs experiments/rwkv7_ks2/gpu_runs/mux_hypotheses
```

Completed independent architectural arms, training seed 11 / coding seed
103415, 15,000 steps:

| Arm | Full-shard validation | Change from raw baseline | 65k validation | Sustained ms/step |
|---|---:|---:|---:|---:|
| Raw-mux baseline, no attention skip | 1.677181 | — | 1.694025 | 14.159 |
| Head residual | 1.676669 | -0.000513 | 1.693405 | 14.119 |
| Learned output gains | 1.679864 | +0.002683 | 1.697183 | 14.024 |
| Persistent KS2 state | 1.676672 | -0.000510 | 1.693205 | 14.169 |
| Separate source normalization | 1.673634 | -0.003547 | 1.690896 | 14.689 |
| Earlier demuxed-history reference | 1.669112 | -0.008070 | 1.687242 | 15.555 |

Source normalization is the strongest of these single-seed architectural
changes, recovering about 44% of the default-point gap to demuxed history.
The head residual and persistent-state improvements are small; learned
output gains initialized at 0.05 do not help in this configuration. These
arms have not each been replicated across training/coding seeds. Timing
uses concurrent jobs on different GPUs, whereas the original reference
timings came from isolated GPU0 runs; sub-percent speed differences should
not be interpreted as established architecture effects.

All 12 new runs completed. The matched reference comparisons are:

| Training seed | Coding seed | Raw-mux loss | Demuxed-history loss | Raw minus demuxed |
|---:|---:|---:|---:|---:|
| 11 | 103415 | 1.677181 | 1.669112 | +0.008070 |
| 22 | 103415 | 1.674290 | 1.666942 | +0.007348 |
| 33 | 103415 | 1.674635 | 1.664470 | +0.010165 |
| 11 | 103416 | 1.679296 | 1.665922 | +0.013374 |
| 11 | 103417 | 1.672513 | 1.669057 | +0.003456 |

Across three training seeds at default coding, raw-mux loss is
1.675369 ± 0.001579 and demuxed-history loss is 1.666841 ± 0.002322
(sample SD). Their mean paired gap is +0.008528 ± 0.001463.
Across three coding seeds at training seed 11, the means are
1.676330 ± 0.003471 and 1.668030 ± 0.001826, with a paired gap of
+0.008300 ± 0.004963. The demuxed-history model wins all five unique
settings. Fixed codes materially affect the gap's size but do not reverse
it in these tests. The source-normalization improvement still needs
replication across codes/training seeds; its one-seed result does not
resolve the entire architecture gap.

All checkpoints reached 15k with finite losses/gradient norms, verified
saved optimizer counters and matching checkpoint reevaluation. The summary
checks source hashes, data manifests, parameter counts, no attention skip,
3.84M training targets per run, and 1,777,024 full-shard target visits with
batch-64 versus batch-4 parity. No experiment combines multiple fixes.

[Hypothesis summary](mux_hypotheses_summary.json) ·
[All fresh and reference results](mux_hypotheses_results.json)

### Corrected single mux at each layer entrance

[`SingleMuxModel`](single_mux.py) implements the user's clarified layout:
one shared input mux per layer produces 64 rank-8 coordinate sets for KS2
and one 96D view for RWKV. The prior raw-mux experiments incorrectly used
two sequential muxes, one before RWKV and another inside the KS2 decoder.
Those results remain preserved as distinct architectures.

```text
history = normalized_embedding
for layer in layers:
    joined = shared_mux_608(LayerNorm(history))  # exactly one F.linear
    coordinates, attention_view = split(joined, [512, 96])
    attention_state = attention_view + RWKV(norm(attention_view))
    coordinates = coordinatewise_causal_time_mix(coordinates)
    raw = coordinates + KS2_geometry(coordinates)  # no down/up projection
    history = concat(history, attention_state, raw)
final_residual = concat(raw, attention_state)
logits = head(norm(final_demux_96(final_residual)))
```

The history contains the embedding and every prior 96D attention state
and 512D raw KS2 output, with the latest previous residual included once.
Input widths are 96 / 704 / 1,312 / 1,920. Attention and KS2 read separate
slices of the same mux output; current-layer attention is not remuxed into
that layer's geometry. The 96 attention channels pass through the decoder
unchanged as part of the layer residual. After clarification about keeping
the final attention contribution, the final demux reads the complete 608D
residual and maps it to the 96D normalized logits input. Consequently the
last RWKV block is connected to the loss. There is no intermediate demux,
no input mux inside any decoder, and no inter-layer attention skip.

KS2 views retain the signed/permuted 8-way Hadamard folding scheme. The
RWKV view uses one 96-way Hadamard matrix, constructed from a verified
48-way sign matrix and doubled to 96. The input is permuted, padded to a
multiple of 96, transformed in 96-way groups, signed, folded and row
normalized. Every real input feature enters every output coordinate.
Padding at input widths 704/1,312 means the 96 output rows are not exactly
orthogonal, but they have unit norm and full row rank. The shared 608-row
buffer concatenates the 512 KS2 rows and 96 RWKV rows and is applied once.
The final fixed 96x608 demux has unit row norms and learned gains initialized
to one, retaining the 5.5x gain LR/inverse weight-decay rule.

There are 540,000 parameters: 269,568 RWKV, 209,920 KS2 geometry, 49,152
embedding/head, 96 final gains, and 11,264 normalization/time-mix parameters.
Shared normalization moves to the layer entrance, and FFN causal time
mixing now acts coordinate-wise on the 512 mux outputs. These changes,
the 96D attention code, and the final 608D readout mean this is a corrected
architecture comparison rather than a single-variable ablation.

[`check_single_mux.py`](check_single_mux.py) checks exactly four shared mux
calls per model forward, no decoder down/up projections, all-input coverage,
unit rows and a full-rank 96D view, exact history/split wiring, paired
attention/geometry initialization, gradient flow through the final RWKV
and every KS2, causal outputs, and GPU recurrence forward/all-gradient
agreement. The runner first re-benchmarks the old two-mux architecture on
GPU0, then benchmarks and trains the new model on the same GPU.

```bash
../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/run_single_mux.py --out experiments/rwkv7_ks2/gpu_runs/single_mux_64
```

Completed seed-11 run at 15,000 steps:

| Architecture | Parameters | Full-shard loss | Benchmark ms/step | Sustained ms/step |
|---|---:|---:|---:|---:|
| Corrected single mux, full residual readout | 540,000 | 1.683107 | 13.397 | 13.967 |
| Previous two-mux raw history | 543,136 | 1.677181 | 13.539 (fresh) | 14.159 |
| Earlier 64-subspace/2,048-side | 564,096 | 1.670096 | 14.753 | 15.419 |
| Earlier demuxed history | 567,552 | 1.669112 | 14.874 | 15.555 |

The first two benchmark times are fresh, sequential 100-update measurements
on physical GPU0. Other timings, including the two-mux sustained time, are
from their earlier completed runs. The corrected model's benchmark is
1.05% faster than the freshly measured two-mux model; sustained training
took 209.498 seconds. Full-shard validation loss is 0.005926 higher than
the two-mux seed-11 result. This is one seed and the architecture also
changes normalization placement, coordinate-wise time mixing, attention
codes and final residual readout, so the loss difference cannot be assigned
solely to the number of mux operations.

All 15k steps had finite losses/gradient norms. Source hashes, optimizer
counters and checkpoint reevaluation passed. Full-shard evaluation visited
1,777,024 targets at four overlapping offsets; batch-64 versus batch-4 loss
differed by at most 9.6e-7.

[Single-mux summary](single_mux_summary.json) · [Single-mux results](single_mux_results.json)


## Single-mux input/output gains and normalized KS2 modulation

These arms extend the corrected single-mux architecture above. All four layers
retain the 64 rank-8 main KS2 subspaces, one 96D RWKV view, existing history,
and final 608-to-96 demux. Each arm is separate:

- **Output gains:** 608 learned scalars per layer after the fixed mux and before
  branch splitting. Unit initialization; 2,432 added parameters, 542,432 total.
- **Input gains:** one scalar per normalized history input, after shared
  LayerNorm and before the fixed mux. Widths 96/704/1312/1920; unit initialization;
  4,032 added parameters, 544,032 total.
- **Dynamic modulation:** append an independent rank-8 view to the same fixed
  mux multiply (616 rows total). An extra geometry-only KS2 cell transforms
  these eight coordinates into eight scalar activations. Each scalar multiplies
  its own learned 608D vector. Sum all eight scaled vectors, normalize that
  combined vector once to unit RMS across its 608 coordinates, and multiply
  it into the main mux outputs before branch splitting. The control subspace
  is local to its layer and is not appended to history; it has no separate
  temporal mix. There are no additional static gains in this arm.

The dynamic computation is `z = KS2(control); g = z @ V;
normalized_g = g / sqrt(max(mean(g*g), 1e-8)); y = main_mux * normalized_g`,
where `V` has shape 8 by 608. There is no additive offset and no individual
vector normalization. Across four layers this adds 19,456 vector weights and
3,280 geometry parameters: **562,736 parameters total**. Vectors start from
Normal(0, 1/sqrt(8)), using seed+3000; the extra fixed view uses seed 103480.
This arm changes the initial function. Original learned weights and all
original 608 mux rows remain paired with the reference.

Static gains and dynamic vectors use the established 5.5x gain LR and
weight decay 0.01/5.5. Extra geometry uses the base LR and weight decay 0.01.
The static arms exactly preserve CPU initial logits, with GPU differences
below 1e-6 in the validation check. The checks also verify the explicit sum
of eight weighted vectors before normalization, unit RMS, nonzero finite
gradients into gains and extra geometry, optimizer updates, and causality.

The sweep trains three gain arms and a fresh unchanged single-mux reference
concurrently across all four MI100s, followed by a fresh demuxed-history
reference. Each uses seed 11, 15,000 updates, batch 4, sequence length 64,
the same training samples and FP32 optimizer schedule. Full-shard evaluation
uses the established four overlapping offsets and 1,777,024 target visits.
This is one training seed; GPU runs are not bit deterministic.

Fresh results (lower loss is better):

| Variant | Parameters | Full-shard loss | Benchmark ms/step | Sustained ms/step |
|---|---:|---:|---:|---:|
| Unchanged single mux | 540,000 | 1.685143 | 13.470 | 13.916 |
| Static output gains | 542,432 | 1.683481 | 13.526 | 13.849 |
| Static input gains | 544,032 | 1.683100 | 13.566 | 14.109 |
| Normalized KS2 modulation | 562,736 | 1.675799 | 16.597 | 17.371 |
| Demuxed-history reference | 567,552 | 1.666869 | 15.302 | 15.637 |

The dynamic arm improves full-shard loss by 0.009344 versus the fresh
single-mux control, recovering about 51% of its gap to the fresh
demuxed-history reference. Sustained steps cost about 25% more than the
unchanged single-mux model. The static arms improve by only 0.001662 and
0.002043. The unchanged single-mux result itself moved by +0.002036 from
its earlier seed-11 run; the demuxed-history repeat moved by -0.002243.
Thus the static gains are inconclusive at this resolution. The dynamic
result is promising for this configuration, but it adds parameters and
changes initialization; there is no replicated-seed claim or isolated
claim that normalization alone caused the improvement.

All 75,000 updates were finite. Source hashes, data manifests, optimizer
counters and checkpoint reevaluation passed for all five runs. The maximum
full-shard batch-4 versus batch-64 loss discrepancy was below 1e-6.

[Gain summary](single_mux_gains_summary.json) · [Gain results](single_mux_gains_results.json)


## RWKV linear-to-mux ablation on the selected demuxed-history baseline

The user selected `HistoryModel(seed, side_width=2048, charts=64)` as the
baseline going forward; the machine-readable selection is
[current_baseline.json](current_baseline.json). All four decoder blocks,
96D main stream, 2048D side stream, separate prior attention/FFN history,
embedding, logits, and KS2 mux/demux remain this baseline throughout the
following comparison.

`rwkv_mux.py` replaces selected `nn.Linear` modules inside RWKV with fixed
signed/permuted Hadamard muxes and learned per-output gains. Every output
coordinate receives every input feature. Codes use independent rank-8
views with unit row norms and full attainable matrix rank. Each gain starts
at the replaced dense row's L2 norm, preserving zero-initialized paths and
input-isotropic output scale. Original learned parameters outside the
replaced matrices are paired exactly. Initial whole-model logits match
exactly on CPU because attention and decoder residual outputs start at zero.
Internal fixed projection orientations differ.

The twelve isolated modes are `read`, `key`, `value`, `output`, and the
`down` and `up` projections of each of `decay_control`, `rate_control`,
`gate_control`, and `value_control`. Each mode replaces that projection
type in every layer where it exists. Value controllers occur only in the
last three layers; other types occur in all four. The all-replaced arm
changes all 46 RWKV linear modules. Controller nonlinearities, recurrence,
normalization, time mixing, biases, key scaling, and residual additions
remain intact. No additional KS2 cells or dynamic gain modulators are added.

Square read/key/value/output projections are 96-to-96. Decay, rate, and
value controllers use 96-to-32 down and 32-to-96 up; the gate controller uses
96-to-64 down and 64-to-96 up. Fixed code seed is `203415 + 100*layer + type_index`,
where `type_index` follows the order in `rwkv_mux.PROJECTIONS`; an isolated
slot and its all-replaced counterpart therefore have identical codes.

RWKV mux gains retain the base LR and weight decay of the dense weights
being replaced. Existing decoder demux gains retain 5.5x LR and inverse
weight decay. Thus only the selected projection parameterization changes;
no optimizer boost is introduced for the new RWKV gains. With all linears
replaced, dense RWKV matrix parameters decrease from 264,192 to 3,584 gain
parameters. The other 5,376 RWKV parameters remain; total model parameters
fall from 567,552 to 306,944. Matrices are precomputed buffers applied by
`F.linear`, so these tests do not use an optimized Hadamard butterfly kernel.

`check_rwkv_mux.py` validates every arm's exact initial function and
unchanged parameters, replacement counts, code coverage/full rank,
per-row scale and zero initialization, optimizer groups, finite gradients,
actual updates after zero residual paths open, and causality. The all-mux
model's GPU recurrence forward and parameter gradients agree with the
reference recurrence.

The fourteen-arm sweep uses four MI100 GPUs, one independent job per GPU.
All runs use seed 11, 15,000 steps, batch 4, 64-token sequences, identical
training samples, and the established FP32 cosine schedule. Full-shard
validation visits 1,777,024 targets over four overlapping offsets, with
batch-4 versus batch-64 parity checked. This is one training seed per arm.

Completed RWKV projection results:

| Replaced projection type | Parameters | Full-shard loss | Delta vs baseline | Sustained ms/step |
|---|---:|---:|---:|---:|
| baseline | 567,552 | 1.665535 | +0.000000 | 15.425 |
| all | 306,944 | 1.767811 | +0.102275 | 15.410 |
| read | 531,072 | 1.677971 | +0.012436 | 15.555 |
| key | 531,072 | 1.683895 | +0.018359 | 15.485 |
| value | 531,072 | 1.681616 | +0.016081 | 15.610 |
| output | 531,072 | 1.688478 | +0.022943 | 16.132 |
| decay_control.down | 555,392 | 1.672065 | +0.006530 | 16.090 |
| decay_control.up | 555,648 | 1.673111 | +0.007576 | 16.134 |
| rate_control.down | 555,392 | 1.669409 | +0.003873 | 15.815 |
| rate_control.up | 555,648 | 1.668936 | +0.003401 | 16.483 |
| gate_control.down | 543,232 | 1.677438 | +0.011903 | 16.725 |
| gate_control.up | 543,360 | 1.665681 | +0.000146 | 16.493 |
| value_control.down | 558,432 | 1.672183 | +0.006648 | 15.998 |
| value_control.up | 558,624 | 1.670819 | +0.005283 | 16.690 |

Replacing every RWKV linear worsens loss by 0.102275 while reducing total
parameters by 260,608 (45.92%). The output projection is the most costly
individual replacement (+0.022943), followed by key (+0.018359) and value
(+0.016081). Gate-controller up is essentially tied (+0.000146) despite
removing 24,192 parameters. Rate controller replacements have smaller
effects than the main projections. These are isolated ablations; combining
individually tolerable replacements was not tested. Small differences need
replication: earlier same-seed dense baseline losses were 1.669112 and
1.666869, versus 1.665535 here. All arms passed source/data/checkpoint
validation and every training update was finite.

[RWKV mux summary](rwkv_mux_summary.json) · [Full results](rwkv_mux_results.json)


## RWKV-8-style RWKV7 + ROSA hybrid on demuxed-history decoders

The follow-up adapts the parallel RWKV7+ROSA architecture in BlinkDL's
[official RWKV-v8 hybrid demo](https://github.com/BlinkDL/RWKV-LM/blob/20be0f8cf0bdf506d8c0030c87a9a3416f95b143/RWKV-v8/251105_reverse_run.py).
The source commit, license, reference functions and hashes are saved under
`source_snapshots/rwkv8_reference/`. This is a local width/budget adaptation
with a supplied training rule, not an official pretrained model or complete
RWKV-8 training recipe.

Each layer computes RWKV and ROSA from the same 96D main residual, with
separate LayerNorms, and adds both updates before the unchanged KS2 decoder.
ROSA has 32 independent 1-bit Q/K/V streams, three 96-to-32 biased input
projections, three learned 96D time-mix vectors, 32 payload scales, and a
biased 32-to-96 output projection. Its hard forward finds the longest query
suffix in strictly previous keys, chooses the latest matching endpoint,
and reads that endpoint's successor value. A matched one gives +payload;
matched zero and no match both give -payload, as in the pinned 1-bit hybrid.
The combined post-attention state is what enters decoder history.

The official hybrid script supplies only ROSA inference. Here, every Q/K/V
single-bit counterfactual credit is computed over the entire 64-token window,
then mapped to logits with `sigmoid(x) * (1-sigmoid(x))`, temperature 1,
following the Bernoulli credit mapping described by
[the exact bit-credit reference](https://github.com/xiaoiecc/qkv-rosa-fast-exact-backward).
This is an explicit discrete surrogate, not the analytic derivative of the
hard threshold. It does not truncate matches within the 64-token window.
The GPU kernel uses packed 64-bit sequences and exact suffix comparisons;
its backward enumerates the bit flips, optimized for this short context.
It is not the asymptotically optimized long-context suffix-automaton kernel.

The budget-matched arm reduces decay/rate/value controller ranks from 32 to
16 and gate rank from 64 to 40, keeping the leading original factor slices.
Adding ROSA and its norms then gives 567,296 parameters, only 256 below the
567,552-parameter baseline. The reduced-controller control without ROSA has
515,328 parameters. A fourth arm adds ROSA while retaining original controller
ranks, totaling 619,520. All four keep the same 96D recurrent attention width,
four 64-rank8 KS2 decoders, 2048D decoder-only side stream, and separate prior
post-attention/post-FFN history. Dense RWKV and ROSA projections remain learned.

ROSA output weights/biases start at zero, payloads at one, and time mixes at
zero. This preserves exact initial CPU logits in every arm. All original
unmodified weights remain paired; ROSA parameters match between hybrid arms.
The optimizer/data/15,000-step protocol matches the earlier experiments.

Validation compares the hard forward against the pinned official suffix
algorithm, checks all 64 bit positions, and compares GPU Q/K/V credits with
brute force at lengths 1, 8, 17 and 64. Maximum gradient error was 4.8e-7.
The core GPU forward+backward benchmark was 0.521 ms for batch4/length64/32
channels. Model checks cover paired state and initial logits, parameter
counts, finite updates and nonzero gradients into every ROSA QKV projection,
and causality. All four arms train concurrently on the four MI100s.

Completed RWKV7+ROSA results:

| Variant | Parameters | Full-shard loss | Delta vs fresh baseline | Benchmark ms/step | Sustained ms/step |
|---|---:|---:|---:|---:|---:|
| Original demuxed-history baseline | 567,552 | 1.666923 | +0.000000 | 14.945 | 15.642 |
| Reduced controllers, no ROSA | 515,328 | 1.675211 | +0.008288 | 15.149 | 15.524 |
| RWKV7+ROSA, matched budget | 567,296 | 1.673763 | +0.006840 | 18.730 | 19.734 |
| RWKV7+ROSA, original controllers | 619,520 | 1.671107 | +0.004184 | 19.322 | 20.144 |

The matched-budget hybrid is 0.006840 worse than the fresh baseline. Adding
ROSA to the reduced-controller model improves loss by only 0.001448, while
adding it at original controller ranks worsens loss by 0.004184. Neither
hybrid beats the selected baseline at this training length and context.
The smaller recovery is comparable to previous same-seed run variation.
These results describe this explicitly documented 1-bit hybrid/backward
adaptation at context64, and do not establish the behavior of a full-scale
RWKV-8 model or its long-context benefits.

All four completed 15,000 finite updates. Source/data/optimizer-counter
and checkpoint restoration checks passed. Full-shard evaluation used the
same 1,777,024 target visits, with batch-4 versus batch-64 discrepancies
below 1e-6. The selected baseline remains unchanged.

[RWKV8 hybrid summary](rwkv8_summary.json) · [Full results](rwkv8_results.json)
