# Capacity-preserving wall-clock study

Planned research window: 2026-10-23 23:44:32 UTC through 2026-10-24 03:44:32 UTC. The user stopped the study early at 2026-10-24 02:38:52 UTC. Active runs and queued launches were terminated; no reserved validation was performed.

## Target and escalation

Beat the original nanoGPT MLP control's validation loss within its measured complete-run time, preserving the KS3 FFN parameter budget. After a three-seed win, retime the unchanged MLP at four times its previous training steps and target that measured loss and duration. The progression is 500 → 2,000 → 8,000 → 32,000 → 128,000 steps, subject to the research window.

The original target is 7.801634 CE in 39.897833 seconds. A later, longer control has its own newly measured target; its time is not extrapolated from the 500-step benchmark.

## Capacity and control

- Original MLP: eleven 96→352→96 ReLU² FFNs, 67,584 learned parameters each, 743,424 collectively.
- KS3 must retain 741,400–743,424 learned FFN parameters collectively. Every parameter remains trainable and participates in computation. No pruning, frozen geometry, or unused padding parameters.
- Primary KS3: 337 rank-1 charts per FFN, 67,400 parameters each, 741,400 collectively. GS32 W8A8 projections, FP16 quantization scales and nonlinearities, FP32 master weights and optimizer state.
- The MLP architecture, initialization, precision, objective, optimizer equations, learning-rate scale 0.25, sequence length 128, and evaluation cadence 100 remain the original control. Only its training horizon increases.
- Configuration and architecture changes apply to KS3 candidates. Mixed ranks retain the same total FFN capacity; longer training sequences change tokens per update, not parameter count.

## Data and evaluation

Use the original FineWeb training and validation shards, verified against the original SHA256 manifest. Training takes one deterministic, EOT-aligned contiguous interval per seed without replay. Seeds are 11, 22, and 33. Increasing training context consumes more tokens per optimizer step; report both quantities.

Screening uses the original 2,048-token validation prefix with **128-token evaluation context**, even when training context increases. Validation uses the entire 50,257-token vocabulary and the original loss/softcap calculation. Training retains the original horizon-relative window, multi-token, prefix, sampled-vocabulary and learning-rate schedules unless explicitly recorded otherwise.

Reserve validation tokens at offset 100,000, length 8,192 for selected finalists. Do not inspect this interval during configuration selection. Checkpoint reloads must first reproduce their original screening loss before interpreting reserved scores.

## Timing

Complete-run timing includes model construction and graph capture, every training step, scheduled validation, and checkpoint saving. Dataset loading, hashing, and disposable warmup pilots are excluded, matching the original GPU benchmark. Each training step is synchronized for timing. Early 650-step screens used evaluation every 130 steps; subsequent runs restore every 100 steps. This difference must remain disclosed.

Four MI100s run independent complete models, one job per GPU. This is experiment parallelism, not tensor-parallel nanoGPT. No overlapping jobs share a GPU during measured training.

Repeated timings of the same seed are averaged within that seed; they are not additional independent quality samples. A reported milestone requires three distinct seeds, each below the target mean runtime and each beating its paired MLP loss. Exploratory single-seed runs are labeled accordingly.

## Metrics and attribution

The current combined metric is **(learned FFN parameters / 743,424) × (complete-run seconds / 39.897833)** on X, versus validation CE on Y. Lower-left is better. The earlier size × step-time metric remains available in the CSV; differing training contexts make step time alone inadequate for the current wall-clock objective.

The control intentionally remains the existing MLP implementation. Acceleration from graph replay, optimizer batching/fusion, local-window attention, and larger training batches is attributed to the complete KS3 training system. These comparisons do not isolate an intrinsic FFN-family speed advantage over an MLP given the same engineering changes.

## Records

Every worker records configuration, seed, GPU, source hashes, data hashes, precision flags, timing, parameter counts, training tokens, validation history, and graph fallback counts. Checkpoints remain local. The report exports completed measurements and launcher outcomes, including failed experiments.

Research checkpoints must be kept with their adjacent `config.json` and loaded through `wallclock_evaluate.load_research_checkpoint`. That loader restores mixed ranks, polynomial basis, attention implementation, precision, and the original Python YaRN scale; ordinary checkpoint loading would miss research-specific behavior.

The pilot model was initially retained by the inspection dictionary while the timed model was constructed. This caused a 4,096-token dense-attention run to exhaust GPU memory during timed setup. Later workers explicitly clear the pilot and collect graph reference cycles before timed setup; subsequent runs record peak allocated GPU memory. This lifetime fix does not change model or optimizer equations.
