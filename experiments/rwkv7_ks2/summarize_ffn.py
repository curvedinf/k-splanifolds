"""Compare new combined KS2 runs with saved controller-ablation results."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

here = Path(__file__).parent
new_root = Path(sys.argv[1])
old = json.loads((here / 'results.json').read_text())
new = [json.loads(p.read_text()) for p in sorted(new_root.glob('ks2_ffn_*.json'))]
assert len(new) == 3
rows = old + new
summary = {}
for variant in ('mlp', 'ks2', 'ks2_ffn'):
    runs = [r for r in rows if r['arguments']['variant'] == variant]
    losses = [r['history'][-1]['val_loss'] for r in runs]
    summary[variant] = dict(parameters=runs[0]['parameters'], controller_parameters=runs[0]['controller_parameters'], ffn_parameters=runs[0].get('ffn_parameters', 262144), loss_mean=statistics.mean(losses), loss_sd=statistics.stdev(losses), training_seconds=statistics.mean(r['training_seconds'] for r in runs), inference_ms=statistics.mean(r['inference_ms'] for r in runs), peak_rss_mib=statistics.mean(r['peak_rss_mib'] for r in runs))
for row in new:
    ref = next(r for r in old if r['arguments']['variant']=='ks2' and r['arguments']['seed']==row['arguments']['seed'])
    assert row['data'] == ref['data']
    for key in ('steps', 'seed', 'threads', 'charts'): assert row['arguments'][key] == ref['arguments'][key]
    assert row['history'][0]['val_loss'] == ref['history'][0]['val_loss']
(here / 'ffn_results.json').write_text(json.dumps(new, indent=2) + '\n')
(here / 'ffn_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
report = '''# RWKV-7: KS2 controllers plus KS2 FFNs

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
'''
labels = {'mlp': 'Original', 'ks2': 'KS2 controllers', 'ks2_ffn': 'KS2 controllers + FFNs'}
for variant, values in summary.items():
    report += f"| {labels[variant]} | {values['parameters']:,} | {values['ffn_parameters']:,} | {values['loss_mean']:.6f} ± {values['loss_sd']:.6f} | {values['training_seconds']:.2f} | {values['inference_ms']:.3f} | {values['peak_rss_mib']:.1f} |\n"
report += '\n| Seed | Original loss | KS2 controllers | KS2 controllers + FFNs | Combined minus controllers |\n|---|---:|---:|---:|---:|\n'
differences = []
for seed in (11, 22, 33):
    losses = [next(r for r in rows if r['arguments']['variant']==v and r['arguments']['seed']==seed)['history'][-1]['val_loss'] for v in labels]
    differences.append(losses[2]-losses[1])
    report += f'| {seed} | {losses[0]:.6f} | {losses[1]:.6f} | {losses[2]:.6f} | {differences[-1]:+.6f} |\n'
combined = summary['ks2_ffn']
report += f"\nCombined minus controller-only mean loss: **{statistics.mean(differences):+.6f} nats/byte**. Combined minus original mean loss: **{combined['loss_mean']-summary['mlp']['loss_mean']:+.6f} nats/byte**. Total parameters decrease **{100*(1-combined['parameters']/summary['mlp']['parameters']):.2f}%** versus original.\n"
report += '''
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
'''
(here / 'FFN_REPORT.md').write_text(report)
sources = [here/'run.py', here/'check.py', Path(__file__)] + list((here.parent/'common').glob('ks2*.py')) + list((here.parent/'common/ks2_cpu').glob('*.py')) + list((here.parent/'common/ks2_cpu').glob('*.cpp'))
(here/'ffn_source_hashes.json').write_text(json.dumps({str(p.relative_to(here.parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2))
print(json.dumps(summary, indent=2))
