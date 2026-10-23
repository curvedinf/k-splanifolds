# Controlled patch denoising

[run.py](run.py) runs MLP, KS1 and reference KS2 on the same CPU denoising task and reports loss. The tables below preserve our previously recorded experiments; they are not results of a new run of this file.

## Citation and scope

Harold C. Burger, Christian J. Schuler and Stefan Harmeling, **Image denoising: Can plain Neural Networks compete with BM3D?**, CVPR 2012. [Paper](https://people.tuebingen.mpg.de/burger/neural_denoising/files/neural_denoising.pdf) · [Authors' project](https://people.tuebingen.mpg.de/burger/neural_denoising/cvpr2012.html).

The control follows the paper's 13×13, two-hidden-layer, 511-unit MLP (Section 5.1/Figure 1), tanh activations, quadratic error, `(pixel−0.5)/0.2` normalization and stride-three patch evaluation. Our CPU adaptation uses Train400/Set12, PyTorch initialization, Adam, a limited step budget, and uniform overlap averaging. The paper uses different training/test corpora, initialization and SGD settings, much longer training, and Gaussian aggregation weights. These results compare models within our adaptation; they do not reproduce the paper's final MLP scores or establish superiority to BM3D.

## Run

From the repository root, with Python, PyTorch, NumPy and Pillow installed:

```bash
python experiments/denoising/run.py --data-dir data/denoising --out runs/denoising
```

Missing images download automatically from [DnCNN](https://github.com/cszn/DnCNN/tree/e93b27812d3ff523a3a79d19e5e50d233d7a8d0a). The immutable revision and SHA-256 digest of the complete sorted image-hash manifest are pinned in the runner. Cached data must pass the same check. Use a new output directory for each invocation; existing directories are never overwritten.

| Default model | Configuration | Parameters |
|---|---|---:|
| MLP | 169 → 511 → 511 → 169; tanh hidden, linear output; biases | 435,030 |
| KS1 | Reference `splanifold_map`; 140 rank-eight charts; learned down/up projections | 434,560 |
| KS2 | Reference `KS2Cell(169,169,123,8,2)` | 433,452 |

KS1 maps projected coordinates through `(1+tanh(c))/2` into the reference chart, adds the resulting displacement to `c`, then projects back to pixels. Its core map is unchanged (`sigma=1`, `extrapolation=0`); the projected residual wrapper is part of this experiment. Its default chart count matches the original MLP/KS2 parameter budget. **The 140-chart KS1 default has not yet had a full benchmark run; historical KS1 results below used 36 charts.**

### Controls

- Grayscale [0,1]; independent Gaussian noise σ=25/255; noisy inputs are not clipped.
- Sorted Train400 filenames; NumPy split seed 20261022 selects 360 training and 40 validation images. Set12 is test-only.
- 120,000 training patches (seed 321), 4,096 validation patches (456), random quarter-turns and horizontal flips; fixed validation noise seed 789.
- Batch 64; 3,000 optimizer steps per model and seed (11,22,33). Identical patch-index/noise streams use a separate generator seeded with `seed+10000`.
- Equal learning-rate search: 0.0001, 0.0003, 0.001, each for 400 steps with seed 777; choose minimum validation MSE. Adam, constant learning rate, gradient norm clipped to one.
- Validate at step one, every 250 steps and the final step. Retain minimum-validation-MSE checkpoint. Evaluate Set12 only after all model selection and training finishes.
- Test noise seed `12345+image_index`, sorted images; reflect padding, 13×13 patches, stride three, uniform overlap averaging, final output clipped to [0,1].
- Eight CPU threads; eager FP32. Inference timing includes normalization, model and concatenation, but excludes patch extraction/reassembly. The runner is a quality control, not a fused-kernel speed benchmark.

The output contains arguments/environment/source hashes, image hashes and exact splits, pilot decisions, training/validation loss histories, best checkpoints, per-image `test_results.csv`, and `summary.json`. Training logs include normalized MSE and its pixel equivalent (×0.04). Validation/test losses use pixel units. Test summary MSE averages image MSEs equally; PSNR averages image PSNRs, so it is not `−10 log10(mean MSE)`. PSNR sample SD is across the three seed means.

A smoke run exercises all models, data checks, checkpoint selection and all twelve test images:

```bash
python experiments/denoising/run.py --out runs/denoising-smoke \
  --steps 2 --pilot-steps 1 --seeds 11 --lrs 0.0003 \
  --train-patches 128 --val-patches 128
```

To use the historical compact KS1 configuration, pass `--ks1-spaces 36 --lrs 0.0003 --steps 1000` (or 3000). This changes the shared learning-rate setting for all three models; it does not reproduce the historical MLP's separately selected 0.001 rate. Defaults retain the original MLP/reference-KS2 protocol. The historical QAT-trained fast KS2 and native inference path are **not** reproduced by this runner; see [the kernel package](../common/ks2_cpu).

## Recorded quality results

All rows use the same image split, noise, patch normalization and reconstruction protocol, with seeds 11,22,33. Original MLP/reference KS2 used equal learning-rate search; compact KS1 used fixed 0.0003, and fast KS2 came from the separate 1,000-step quantization-aware experiment. Budgets, sizes and training procedures differ across those groups.

| Recorded model | Parameters | Main steps | Mean image MSE¹ | Mean PSNR (dB) | Seed SD (dB) |
|---|---:|---:|---:|---:|---:|
| MLP (original) | 435,030 | 3,000 | 0.002399074 | 26.302535 | 0.113845 |
| KS2 (reference) | 433,452 | 3,000 | 0.001871274 | 27.388240 | 0.014819 |
| KS1 (36 charts, 1000 steps) | 111,744 | 1,000 | 0.002288613 | 26.542287 | 0.005206 |
| KS1 (36 charts, 3000 steps) | 111,744 | 3,000 | 0.002299040 | 26.474445 | 0.008101 |
| KS2 (fast stabilized Horner) | 112,768 | 1,000 | 0.002210024 | 26.714944 | 0.011888 |

¹ Historical image MSE is recovered **per image and seed** as `10**(-PSNR/10)` from the saved image PSNRs, then averaged; it is not obtained by exponentiating the aggregate PSNR. Small rounding differences from directly saved loss are possible. Fresh runner output records MSE directly.

## Recorded inference speed

Same CPU environment, eight threads; average milliseconds per Set12 image. Compiled/fused measurements exclude compilation and use three timing passes. These were separate historical timing sessions, not one interleaved head-to-head rerun. Hardware was an AMD EPYC 9V74 host under an eight-CPU quota; PyTorch 2.14.1+cpu.

| Model/backend | Training steps | Batch 128 (ms/image) | Batch 512 (ms/image) |
|---|---:|---:|---:|
| MLP, torch.compile default | 3,000 | 138.191 | — |
| KS1, torch.compile, 36 charts | 1,000 | 68.172 | 44.156 |
| KS1, torch.compile, 36 charts | 3,000 | 69.682 | 44.833 |
| KS2 stabilized, FP32 AVX-512 Horner | 1,000 QAT | 53.986 | 35.819 |

The fast KS2 row is the FP32 Horner export of the compact trained model, not INT8 VNNI. Its reported PSNR is 26.714944. The original unmodified 433,452-parameter KS2 reached 27.388240, but had a different size and training budget. Timing alone does not compare convergence or total training cost.

## Image-level audit

Mean PSNR across the three seeds, before rounding for display. MSE above was recovered from individual seed/image values, not these averages.

| Set12 image | MLP original | KS2 reference | KS1 1000 | KS1 3000 | KS2 fast |
|---|---:|---:|---:|---:|---:|
| 01 | 25.267389 | 26.793059 | 25.439404 | 25.592492 | 25.570130 |
| 02 | 27.695705 | 29.018145 | 28.159978 | 27.761166 | 28.513644 |
| 03 | 26.130243 | 27.110552 | 26.174806 | 26.178042 | 26.281615 |
| 04 | 26.391108 | 26.983213 | 26.519683 | 26.486725 | 26.630140 |
| 05 | 26.005050 | 26.935545 | 26.083861 | 26.153025 | 26.238082 |
| 06 | 25.573288 | 26.261557 | 25.374123 | 25.497351 | 25.447823 |
| 07 | 24.899254 | 27.089790 | 25.571027 | 25.752190 | 25.771266 |
| 08 | 27.726758 | 29.145435 | 28.281988 | 27.853174 | 28.628874 |
| 09 | 25.040759 | 25.657642 | 24.800673 | 25.019306 | 24.815883 |
| 10 | 26.862415 | 27.871346 | 27.269907 | 27.052562 | 27.466191 |
| 11 | 27.248919 | 28.041534 | 27.537388 | 27.270594 | 27.754502 |
| 12 | 26.789529 | 27.751059 | 27.294601 | 27.076713 | 27.461171 |

Recorded source files from the experiment archive (SHA-256; not dependencies of the runner):

| Archive path | SHA-256 |
|---|---|
| `results/test_results.csv` | `de47b5ac5bd606cf30af3a652ea41ddc4cc0bc1d639c6c3bfbce107300391cdb` |
| `ks1_denoising_results/test.json` | `4ac5125b1c98a2653b9b2b157781578c6cbc4a9a544d02f75e7b3cee33ca138b` |
| `optimization_results/timings.json` | `894f7e13bead178957043943e6763506c5ac45433d7d17a49d6e2e376b24f1e5` |
| `mlp_compile_default_results/summary.json` | `830b7e84b1ffba99813cf497c5248a76a32b37ac032e57618fbd478a52308c01` |
| `optimization_batch_results/summary.json` | `fe283f02606c21e80de4c1522d0233aec1860c52689b942b8f61fde726c9945f` |

## All-weight GS128 ternary QAT, 3,000 steps

New runs train the compact stabilized KS2 (32 charts, rank eight, one geometry step; 112,768 parameters) from scratch for 3,000 steps, seeds 11/22/33. All learned weights, including both projections, geometry, damping and warps, are quantized each forward using the custom padded GS128 TQ1-style format tested in nanoGPT. Each 128-weight block uses 26 base-three bytes plus one FP16 absmax scale. This is not the standard llama.cpp TQ1_0 ABI. FP32 master weights receive identity straight-through gradients; activations, optimizer states, and analytic fused backward remain FP32. The quantizer uses original per-chart geometry packing and per-row projections; warp vectors group across charts per family.

The original denoising dataset hash, split, patch/noise streams, normalization, batch 64, Adam learning rate 0.0003 and gradient clipping 1 are retained. Packed-model initialization draw order matches the historical compact experiment. No learning-rate search or controls were rerun. This differs from historical fast KS2's INT8 inner arithmetic / FP16 geometry and unquantized projections: the new test specifically isolates all-weight ternary QAT with FP32 arithmetic.

Fixed checkpoints at steps 1,000 and 3,000 were evaluated after all training finished, with no best-validation checkpoint substitution. Set12 reconstruction is unchanged.

| Model/checkpoint | Validation patch MSE | Set12 image MSE | Mean Set12 PSNR | Seed SD |
|---|---:|---:|---:|---:|
| TQ1 GS128 QAT, 1000 steps | 0.004986039 | 0.002404623 | 26.383547 dB | 0.058557 dB |
| TQ1 GS128 QAT, 3000 steps | 0.005231583 | 0.002370954 | 26.436996 dB | 0.230287 dB |
| Recorded MLP, 3,000 steps | — | 0.002399074 | 26.302535 dB | 0.113845 dB |
| Recorded fast KS2, 1,000 steps | — | 0.002210024 | 26.714944 dB | 0.011888 dB |

| Seed | Ternary 1,000-step Set12 PSNR | Ternary 3,000-step Set12 PSNR |
|---|---:|---:|
| 11 | 26.325653 | 26.257372 |
| 22 | 26.382241 | 26.696612 |
| 33 | 26.442746 | 26.357005 |

At 3,000 steps, packed weights occupy **31,080 bytes**, including scales and padding, versus 1,740,120 bytes for the recorded 435,030-parameter FP32 denoising MLP (about56× smaller). FP32 master KS2 weights occupy 451,072 bytes. Packed storage excludes metadata and is not resident QAT memory. Actual packed export exactly matched all fake-quantized parameter tensors; the packed and QAT validation losses agreed within 1e-8. Fused versus reference stabilized output at the final seed11 checkpoint differed by at most 7.16e-7 over 128 noisy patches. All training losses and gradient norms remained finite.

Continued training yielded only +0.05345 dB on average; two seeds became worse and one improved. Mean validation MSE rose from 0.004986039 to 0.005231583. The +0.13446 dB final mean versus recorded MLP is small relative to seed variation; this does not establish a reliable quality advantage. The earlier fast KS2 result remains better at 26.714944 dB.

```bash
python experiments/common/ks2_cpu/build.py
python experiments/denoising/train_tq1.py \
  --data-dir data/denoising --out runs/denoising-tq1 \
  --steps 3000 --seeds 11 22 33
```

The companion archive includes source dependencies, configuration and hashes, complete training/validation histories, per-image Set12 results, export checks, and all six FP32-master checkpoints. Restore a checkpoint with `train_tq1.model(seed,threads)` and `load_state_dict`; use `train_tq1.packed(model)` to convert to decoded deployment weights and disable fake quantization. QAT does not quantize optimizer state or backpropagation.

The fused training backend requires an AVX-512 CPU, a C++ compiler with OpenMP, and libmvec. Build the kernels before running the command above. Python dependencies include PyTorch, NumPy, Pillow and tiktoken (imported by the shared nanoGPT helper module dependencies).

Raw ternary QAT results and histories are committed in [`tq1_results/`](tq1_results/). Earlier FP16/FP8/INT8 numerical and training experiments are preserved in [`quantization/`](quantization/README.md). Checkpoints and datasets are generated by the runners and are not committed.
