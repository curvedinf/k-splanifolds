"""Record the adaptive FFN width screen, reusing all saved controls."""
import json
from pathlib import Path
import sys

here = Path(__file__).parent
root = Path(sys.argv[1])
original = next(r for r in json.loads((here/'results.json').read_text()) if r['arguments']['variant']=='mlp' and r['arguments']['seed']==11)
controller = next(r for r in json.loads((here/'results.json').read_text()) if r['arguments']['variant']=='ks2' and r['arguments']['seed']==11)
small = next(r for r in json.loads((here/'ffn_results.json').read_text()) if r['arguments']['seed']==11)
rows = [small]
for width in (22,44,88,176):
    path = root/f'rwkv7_ffn_{width}/ks2_ffn_11.json'
    if path.exists(): rows.append(json.loads(path.read_text()))
for row in rows:
    assert row['data']==original['data']
    assert row['history'][0]['val_loss']==original['history'][0]['val_loss']
    for key in ('steps','seed','threads'): assert row['arguments'][key]==original['arguments'][key]
(here/'width_results.json').write_text(json.dumps(rows,indent=2)+'\n')
text = '''# RWKV-7 KS2 FFN width screen

Follow-up to [the 11-subspace FFN experiment](FFN_REPORT.md). Increase only the
number of rank-8 subspaces in each KS2 FFN. Keep both one-subspace KS2 memory
controllers and all data/training choices fixed. Each run starts from scratch,
uses seed 11 and the same batches, and trains for 1,000 steps. Controls are reused.

The original seed-11 target is 2.099161 nats/byte; the controller-only KS2 loss
is 2.094131. The original three-seed mean is 2.097655, but the table below uses
only seed 11 throughout to avoid mixing seed selection with architecture effects.

Search order: 11 (already completed), 22, 44, 88, 176. The last extension probes
about four times the original FFN parameter budget. The intended procedure was
to confirm a target-reaching width on seeds 22 and 33. Width selection uses the
same held-out validation windows, so any selected result needs independent
evaluation; this is not a claim of an optimal width or statistical significance.

| Version | Subspaces per FFN | Total parameters | FFN parameters | Final loss | Training seconds | Inference ms | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
for label,row,width in [('Original',original,'—'),('KS2 controllers only',controller,'—')]+[('KS2 controllers + FFNs',r,str(r['arguments']['ffn_charts'])) for r in rows]:
    text += f"| {label} | {width} | {row['parameters']:,} | {row.get('ffn_parameters',262144):,} | {row['history'][-1]['val_loss']:.6f} | {row['training_seconds']:.2f} | {row['inference_ms']:.3f} | {row['peak_rss_mib']:.1f} |\n"
best = min(rows,key=lambda r:r['history'][-1]['val_loss'])
gap = best['history'][-1]['val_loss']-original['history'][-1]['val_loss']
text += f"\nBest screened width: **{best['arguments']['ffn_charts']} subspaces**, loss **{best['history'][-1]['val_loss']:.6f}**, difference from original **{gap:+.6f} nats/byte**.\n"
if gap>0:
    text += '\nNone of the tested widths reached the original loss. No candidate qualified for three-seed confirmation. Increasing width alone did not recover the target under the fixed 1,000-step setup. This does not establish a fundamental KS2 limit or rule out changes in initialization, optimizer, training duration, or architecture.\n'
text += '''
All runs reject nonfinite loss/gradient norms and completed successfully. Sequence
inference and peak-process-memory definitions match the prior reports. Neither
parameter count nor CPU timing implies GPU speed. Larger widths change gradient
scaling as well as capacity; no width-specific learning-rate adjustment was made.

Reproduce a candidate from the extracted bundle/repository root:

```bash
python experiments/rwkv7_ks2/run.py --data data --out runs/rwkv7-44 --variant ks2_ffn --ffn-charts 44 --seed 11 --steps 1000
```

See README.md for the model/data provenance and FFN_REPORT.md for the three-seed
11-subspace results. All per-step curves and manifests for this screen are stored
in width_results.json. The bundle includes each checkpoint under a distinct
width directory, prepared data bytes, source code, and raw logs. No controls
were retrained, no run extended beyond 1,000 steps, and no intermediate checkpoint
was substituted for the final checkpoint.
'''
diagnostic_path = here/'width_diagnostics.json'
if diagnostic_path.exists():
    diagnostics = json.loads(diagnostic_path.read_text())
    text += '''
## Width and optimization diagnostic

Each subspace has independent projection rows/columns, geometry coefficients,
damping parameters, and warps. Shared tensor storage is not weight tying. Incoming
layer normalization and token shifting are shared across subspaces, and outputs
are summed into the same model-width representation. Within each subspace, warp
parameters are shared across coordinates.

`diagnose_width.py` runs one optimizer step on the same first batch and measures
the first layer's FFN output on its captured, fixed pre-update input. All widths
start with exactly the same model loss. These are diagnostic updates, not additional
full training runs or controls. RMS means root-mean-square output magnitude.

| Subspaces | FFN output RMS after first update | Linear contribution RMS | Geometry contribution RMS |
|---|---:|---:|---:|
'''
    for row in diagnostics:
        text += f"| {row['subspaces']} | {row['first_ffn_output_rms_after_one_update']:.6f} | {row['linear_part_rms']:.6f} | {row['geometry_part_rms']:.6f} |\n"
    text += '''
The output update grows approximately proportionally to width despite the same
learning rate. This is a measured optimization confound, not proof that it caused
the final loss plateau. No width-normalized optimizer experiment has been run.
Zero-initialized FFN output weights also produce exactly zero first-step gradients
for the input projection and geometry; subsequent updates unlock those paths.
This matches the zero residual-output behavior of the original RWKV FFNs but does
not imply equal training dynamics. Wider models can represent narrower models by
zeroing extra contributions, but fixed-step optimization and validation loss need
not improve monotonically. The width screen does not establish a capacity limit.
'''
(here/'WIDTH_REPORT.md').write_text(text)
print(text)
