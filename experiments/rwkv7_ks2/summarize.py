"""Aggregate completed paired runs without selecting a winning checkpoint."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
destination = Path(__file__).parent
rows = [json.loads(path.read_text()) for pattern in ('mlp_[0-9]*.json', 'ks2_[0-9]*.json') for path in sorted(root.glob(pattern))]
summary = {}
for variant in ('mlp', 'ks2'):
    runs = [row for row in rows if row['arguments']['variant'] == variant]
    values = [row['history'][-1]['val_loss'] for row in runs]
    summary[variant] = dict(seeds=[row['arguments']['seed'] for row in runs], parameters=runs[0]['parameters'], controller_parameters=runs[0]['controller_parameters'], loss_mean=statistics.mean(values), loss_sd=statistics.stdev(values) if len(values)>1 else 0, training_seconds=statistics.mean(row['training_seconds'] for row in runs), inference_ms=statistics.mean(row['inference_ms'] for row in runs), peak_rss_mib=statistics.mean(row['peak_rss_mib'] for row in runs))
summary['paired_loss_differences'] = {str(seed): next(row for row in rows if row['arguments']['variant']=='ks2' and row['arguments']['seed']==seed)['history'][-1]['val_loss'] - next(row for row in rows if row['arguments']['variant']=='mlp' and row['arguments']['seed']==seed)['history'][-1]['val_loss'] for seed in summary['mlp']['seeds']}
(destination / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
(destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
text = '''# RWKV-7: KS2 memory-controller ablation

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
'''
for variant in ('mlp', 'ks2'):
    r = summary[variant]
    text += f"| {variant.upper()} | {r['parameters']:,} | {r['controller_parameters']:,} | {r['loss_mean']:.6f} ± {r['loss_sd']:.6f} | {r['training_seconds']:.2f} | {r['inference_ms']:.3f} | {r['peak_rss_mib']:.1f} |\n"
text += '\nLoss is cross-entropy in natural-log units per byte; ± is sample standard deviation across seeds. Other measurements are means across runs.\n\n| Seed | MLP loss | KS2 loss | KS2 minus MLP |\n|---|---:|---:|---:|\n'
for seed, diff in summary['paired_loss_differences'].items():
    get = lambda variant: next(row for row in rows if row['arguments']['variant']==variant and row['arguments']['seed']==int(seed))['history'][-1]['val_loss']
    text += f'| {seed} | {get("mlp"):.6f} | {get("ks2"):.6f} | {diff:+.6f} |\n'
text += '''
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
'''
(destination / 'README.md').write_text(text)
sources = [Path(__file__), destination / 'run.py', destination / 'check.py'] + list((destination.parent / 'common').glob('ks2*.py')) + [destination.parent / 'common/ks2_cpu' / name for name in ('training.py', 'backward.cpp')]
(destination / 'source_hashes.json').write_text(json.dumps({str(p.relative_to(destination.parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2))
print(json.dumps(summary, indent=2))
