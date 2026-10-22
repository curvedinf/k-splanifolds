# Controlling the KS2 FFN update scale across widths

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
| 11 | 1.0000 | 0.011413 | 0.011413 |
| 22 | 0.5000 | 0.021850 | 0.010925 |
| 44 | 0.2500 | 0.044306 | 0.011077 |
| 88 | 0.1250 | 0.088284 | 0.011036 |
| 176 | 0.0625 | 0.175129 | 0.010946 |

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
| 11 | 317,512 | 2.171209 | 2.171209 | +0.000000 | 96.15 | 14.833 | 460.3 |
| 22 | 380,608 | 2.174487 | 2.158192 | -0.016295 | 88.67 | 12.258 | 460.5 |
| 44 | 506,800 | 2.167567 | 2.168962 | +0.001396 | 98.58 | 12.325 | 460.2 |
| 88 | 759,184 | 2.182448 | 2.169936 | -0.012512 | 107.70 | 15.911 | 460.2 |
| 176 | 1,263,952 | 2.177584 | 2.158622 | -0.018962 | 127.33 | 19.700 | 470.0 |

Best controlled width: **22**, loss **2.158192**.

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
