"""Compare width-scaled FFN output updates with existing fixed-rate runs."""
import hashlib
import json
from pathlib import Path
import sys

here = Path(__file__).parent
root = Path(sys.argv[1])
old = json.loads((here/'width_results.json').read_text())
diagnostics = json.loads((here/'controlled_width_diagnostics.json').read_text())
prior_diagnostics = json.loads((here/'width_diagnostics.json').read_text())
rows = [next(row for row in old if row['arguments']['ffn_charts']==11)]
for width in (22,44,88,176):
    row=json.loads((root/f'rwkv7_scaled_{width}/ks2_ffn_11.json').read_text())
    prior=next(r for r in old if r['arguments']['ffn_charts']==width)
    assert row['arguments']['control_update_scale']
    assert row['data']==prior['data']
    assert row['history'][0]['val_loss']==prior['history'][0]['val_loss']
    for key in ('steps','seed','threads','ffn_charts','charts'): assert row['arguments'][key]==prior['arguments'][key]
    rows.append(row)
(here/'controlled_width_results.json').write_text(json.dumps(rows,indent=2)+'\n')
text='''# Controlling the KS2 FFN update scale across widths

Follow-up to [the width screen](WIDTH_REPORT.md). Four new seed-11 runs at
22/44/88/176 subspaces, 1,000 steps each; all controls and the 11-subspace reference
are reused. No architecture, initialization, data, minibatch, precision, gradient
clipping, or training-duration changes. The sole optimizer intervention is on
KS2 FFN output-projection parameters.

## Intervention

Multiply the FFN output-projection learning rate by 11 divided by its subspace
count, throughout the cosine schedule. This anchors the update scale to the
original 11-subspace experiment. Input projections, geometry, memory controllers,
and the rest of the model keep their original learning rates.

To preserve AdamW's original multiplicative parameter shrinkage, divide the
output-projection weight-decay coefficient by the same factor. Thus the product
of learning rate and weight decay is unchanged at every step. Adam's moment
updates, epsilon and betas stay unchanged. This is a width-aware learning-rate
rule, not a feedback controller enforcing equal output changes at every step.

## First-update verification

Same first training batch, seed, fixed captured input to the first layer's FFN,
and zero initial output projection. RMS denotes root-mean-square output magnitude.

| Subspaces | Output-projection LR multiplier | Previous first update RMS | Controlled first update RMS |
|---|---:|---:|---:|
'''
for d in diagnostics:
    width=d['subspaces']
    previous=next(p for p in prior_diagnostics if p['subspaces']==width)
    text+=f"| {width} | {11/width:.4f} | {previous['first_ffn_output_rms_after_one_update']:.6f} | {d['first_ffn_output_rms_after_one_update']:.6f} |\n"
text+='''
This verifies removal of the approximately linear growth in the first output
update. It does not guarantee exactly equal updates later in training, or equal
optimization of geometry and projections. Geometry still receives zero gradient
on the first step because the output projection starts at zero; that initialization
is unchanged to isolate this intervention.

## Final validation results

Same 8,192 held-out byte targets, final 1,000-step checkpoint, seed 11. Original
RWKV-7 loss is 2.099161; KS2-controller-only loss is 2.094131. Losses are nats/byte.
The 11-subspace row is reused because the multiplier there is exactly one.

| Subspaces | Total parameters | Previous loss | Controlled loss | Difference | Training seconds | Inference ms | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
for row in rows:
    width=row['arguments']['ffn_charts']
    before=next(r for r in old if r['arguments']['ffn_charts']==width)['history'][-1]['val_loss']
    after=row['history'][-1]['val_loss']
    text+=f"| {width} | {row['parameters']:,} | {before:.6f} | {after:.6f} | {after-before:+.6f} | {row['training_seconds']:.2f} | {row['inference_ms']:.3f} | {row['peak_rss_mib']:.1f} |\n"
best=min(rows,key=lambda r:r['history'][-1]['val_loss'])
text+=f"\nBest controlled width: **{best['arguments']['ffn_charts']}**, loss **{best['history'][-1]['val_loss']:.6f}**.\n"
text+='''
This is a single-seed, adaptively explored validation comparison, not an independent
test-set result. The intervention controls a measured first-update confound; it
does not by itself establish the reason for the earlier convergence plateau.
See prior reports for CPU timing/RSS definitions, architecture, licenses and data
provenance. No existing controls were retrained.

## Reproduce

```bash
python experiments/common/ks2_cpu/build.py
for width in 22 44 88 176; do
  python experiments/rwkv7_ks2/run.py --data data --out runs/scaled-$width --variant ks2_ffn --ffn-charts "$width" --control-update-scale --seed 11 --steps 1000
done
```

Prepared bytes in the bundle match the original data manifests. Recorded runs used
the original shard-loading route (including tokenizer memory); prepared bytes give
identical training data but a different peak process RSS. New checkpoints are under
`checkpoints/rwkv7_scaled_<width>/`. Full curves/configurations are in
controlled_width_results.json and diagnostic values in controlled_width_diagnostics.json.
'''
(here/'CONTROLLED_WIDTH_REPORT.md').write_text(text)
sources=list(here.glob('*.py'))+list((here.parent/'common').glob('ks2*.py'))+list((here.parent/'common/ks2_cpu').glob('*.py'))+list((here.parent/'common/ks2_cpu').glob('*.cpp'))
(here/'controlled_source_hashes.json').write_text(json.dumps({str(p.relative_to(here.parent.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2))
print(text)
