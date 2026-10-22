# RWKV-7 KS2 FFN width screen

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
| Original | — | 537,856 | 262,144 | 2.099161 | 87.08 | 12.015 | 460.1 |
| KS2 controllers only | — | 516,560 | 262,144 | 2.094131 | 90.68 | 14.587 | 460.2 |
| KS2 controllers + FFNs | 11 | 317,512 | 63,096 | 2.171209 | 96.15 | 14.833 | 460.3 |
| KS2 controllers + FFNs | 22 | 380,608 | 126,192 | 2.174487 | 92.98 | 12.051 | 460.3 |
| KS2 controllers + FFNs | 44 | 506,800 | 252,384 | 2.167567 | 102.41 | 14.718 | 460.4 |
| KS2 controllers + FFNs | 88 | 759,184 | 504,768 | 2.182448 | 108.31 | 20.101 | 460.3 |
| KS2 controllers + FFNs | 176 | 1,263,952 | 1,009,536 | 2.177584 | 132.19 | 16.828 | 469.0 |

Best screened width: **44 subspaces**, loss **2.167567**, difference from original **+0.068405 nats/byte**.

None of the tested widths reached the original loss. No candidate qualified for three-seed confirmation. Increasing width alone did not recover the target under the fixed 1,000-step setup. This does not establish a fundamental KS2 limit or rule out changes in initialization, optimizer, training duration, or architecture.

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
| 11 | 0.011413 | 0.010340 | 0.001503 |
| 22 | 0.021850 | 0.019877 | 0.002674 |
| 44 | 0.044306 | 0.040297 | 0.005242 |
| 88 | 0.088284 | 0.080843 | 0.009423 |
| 176 | 0.175129 | 0.159225 | 0.020185 |

The output update grows approximately proportionally to width despite the same
learning rate. This is a measured optimization confound, not proof that it caused
the final loss plateau. No width-normalized optimizer experiment has been run.
Zero-initialized FFN output weights also produce exactly zero first-step gradients
for the input projection and geometry; subsequent updates unlock those paths.
This matches the zero residual-output behavior of the original RWKV FFNs but does
not imply equal training dynamics. Wider models can represent narrower models by
zeroing extra contributions, but fixed-step optimization and validation loss need
not improve monotonically. The width screen does not establish a capacity limit.
