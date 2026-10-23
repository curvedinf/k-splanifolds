# Rank-1 KS2: integrated INT8, Horner, fusion and tensor parallelism

One integrated inference export of the strongest replicated KS2 model: four layers, a 96D residual stream, and 368 rank-1 spine charts per FFN. The original trained budget remains **73,600 parameters per FFN**. The three precision settings below change KS2 nonlinear evaluation, not the architecture or training. All learned linear maps, embeddings and KS2 coefficients are INT8. **Recurrent state stays FP32 in every setting.**

## Quality on the earlier reserved interval

The three original 100k checkpoints (seeds 11, 22, 33; larger training pool) are exported without retraining. Loss is nats/byte, using four overlapping T64 tilings of the same reserved interval used by the previous report. The unchanged matched MLP mean is **1.463041**.

| Implementation | Mean loss | Seed SD | Change vs original KS2 | Change vs MLP | Wins vs MLP |
| --- | --- | --- | --- | --- | --- |
| Original FP32 KS2 | 1.457559 | 0.001409 | +0.000000 | -0.005482 | 3/3 |
| W8A8 + FP32 nonlinearities | 1.457705 | 0.001393 | +0.000146 | -0.005336 | 3/3 |
| W8A8 + FP16 nonlinearities | 1.457696 | 0.001392 | +0.000137 | -0.005345 | 3/3 |
| W8A8 + INT8 nonlinearities | 1.458469 | 0.001362 | +0.000909 | -0.004573 | 3/3 |

INT8 nonlinearities retain the sub-MLP loss result. FP16 is effectively tied with FP32 here; INT8 has a small additional loss cost. FP16 nonlinearities are the hardcoded default, as requested by the user. The reported benchmark settings were selected explicitly and their measurements are unchanged. This default still uses INT8 weights, activations and integer matrix dots.

## Measured inference latency

| Batch × context | Original FP32, TP1 ms | W8A8 TP1 ms | W8A8 TP4, FP32 nonlinear ms | TP4, FP16 nonlinear ms | TP4, INT8 nonlinear ms | TP4 / original speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 4 × 64 | 1.533 | 1.332 | 1.483 | 1.489 | 1.502 | 1.03× |
| 64 × 64 | 5.319 | 4.368 | 3.927 | 3.956 | 3.966 | 1.35× |
| 256 × 64 | 21.526 | 17.214 | 15.100 | 15.383 | 15.742 | 1.43× |

These are captured **forward + cross-entropy evaluation** times, not optimizer steps, autoregressive serving/decode latency, or the earlier ~8 ms training timings. Each sequence starts with zero recurrent state. Five consecutive rounds of 100 graph replays exclude compilation, batch copies and process startup; TP4 records the slowest rank. Cases are not randomized and clocks are not locked, so small differences should not be overinterpreted. Raw rounds remain in the JSON records.

The optimized single-GPU path remains preferable for small batches. Head sharding makes TP4 useful at larger batches; communication and replicated work prevent ideal fourfold scaling. Four GPUs accelerate one model here, rather than running independent experiments.

## What is fused and quantized

- Input projection uses native INT8 × INT8 → INT32 matrix dots. Its epilogue evaluates the rank-1 Hermite polynomial in Horner form, adds the coordinate bypass, applies ReLU², and directly requantizes for the output projection.
- Hermite coefficients are quantized from the trained checkpoint. Horner coefficient combinations happen inside the kernel; this is not a retrained power-basis model.
- Small RWKV controllers fuse their input projection, tanh/sigmoid (or identity), and activation quantization.
- Linear weights use symmetric per-output/group INT8 scales. Activation quantization uses per-token/group scales and round-to-nearest. FFN output groups are 92 features so TP1 and TP4 partition the same quantization groups. Attention output groups are 32 features, aligned with heads.
- RWKV inference omits the backward-only saved-state history while keeping the live recurrent matrix and arithmetic FP32.

The design draws on the repository’s [VNNI KS2 implementation](https://github.com/curvedinf/k-splanifolds/blob/main/experiments/common/ks2_cpu/vnni.cpp): prepacked quantized coefficients, integer accumulation, explicit scales, and floating-point nonlinear/state exceptions. The MI100 round-to-nearest and custom-all-reduce patterns come from `../int8-vllm`; its installed gfx908 collective uses uncached staging for coherent graph replay. No serving configuration or sibling repository was modified.

## Nonlinear precision scope

- **FP32:** fused Horner evaluation and bounded tanh in FP32.
- **FP16:** the KS2 nonlinear inputs, Horner stages and ReLU² intermediates round to FP16; tanh itself uses an FP32 transcendental internally.
- **INT8:** quantized tanh inputs/outputs with a 256-entry table, INT8 spine coefficients, signed INT32 Horner accumulation with six fractional guard bits, and integer ReLU² before requantization. The tanh table covers ±3.2 because its INT8 output already saturates near that range.

The nonlinear precision switch covers **KS2 geometry and ReLU²**. RWKV controller nonlinearities, normalization, time mixing, scales, residual streams, FP32 collective partials, and recurrent state retain FP32. This is the VNNI-style INT8 contract requested by the user, not an integer-only recurrent network.

## Tensor-parallel layout

Each GPU owns **92 of the 368 FFN subspaces** in every layer. The unchanged model has three RWKV heads, so attention heads are assigned **[1, 1, 1, 0]** across ranks. The fourth rank still owns its FFN shard and participates in reductions. Per-head controller output rows and the attention output projection are sharded. Embeddings, head, layer norms, and some small controller input projections are replicated. Both attention and FFN output partials are summed into the 96D residual.

RCCL is a supported fallback. The reported TP4 timings use the existing vLLM MI100 custom all-reduce with an 8 MiB envelope. The initial FFN-only TP split was slower than optimized TP1 at B64; adding head sharding resolved that limitation.

## Checks and artifacts

Independent operator checks compare W8A8 products against explicit integer-operand reference products and fused Horner against basis evaluation. The integer nonlinear path matches its independent signed-integer reference. Compiled assembly contains native INT8 dot instructions. Attention/FFN partition checks, graph-versus-eager checks, all three trained checkpoints and all four ranks pass.

Floating-point reduction order can change an INT8 rounding decision downstream, so complete TP1 and TP4 logits are not bitwise identical. The component checks bound the summation error, and the full held-out scores evaluate the resulting model behavior. Export reload checks require exact state tensors and exact logits for every rank and all three nonlinear settings.

- [Summary and raw timing/quality values](summary.json)
- [Independent operator checks](operator_checks.json)
- [Four-shard reload checks](export_reload_check.json)
- [Export manifest, sizes and hashes](export_manifest.json)
- [Combined kernel implementation](code/rank1_int8_tp.py)
- [Head-sharded attention](code/rank1_int8_attention_tp.py)
- [Export/load helper](code/export_rank1_int8_tp.py)
- [Exact current implementation sources](source_bundle.tar.gz)

The loadable weights remain in the shared workspace at `experiments/rwkv7_ks2/gpu_runs/rank1_int8_tp/export_tp4_seed11/`. Checkpoints and validation text are not published in this report. The source archive extracts into a fresh directory with the original package paths and includes the model builder dependencies. Use the existing ROCm environment; custom collective timings additionally require the tested vLLM MI100 extension.

```bash
# Export the trained checkpoint into four INT8 shards.
python experiments/rwkv7_ks2/export_rank1_int8_tp.py \
  --checkpoint-run PATH_TO_TRAINED_RUN --out NEW_EXPORT_DIRECTORY --tp 4

# Check the integrated implementation against a trained run.
TORCH_NCCL_ASYNC_ERROR_HANDLING=0 torchrun --standalone --nproc-per-node=4 \
  experiments/rwkv7_ks2/check_rank1_int8_tp.py \
  --checkpoint-run PATH_TO_TRAINED_RUN --out NEW_RESULTS_DIRECTORY \
  --batch 64 --collective custom --full-eval --validation VALIDATION_BYTES
```
