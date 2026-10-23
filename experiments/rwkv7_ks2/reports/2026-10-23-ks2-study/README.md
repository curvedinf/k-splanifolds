# KS2 language-model study — 23 October 2026

[Read the full report on GitHub](REPORT.md). It includes all 60 sections, tables and figures. The [standalone HTML version](report.html) is also available.

The study completed **428 training runs**, **259 fresh reserved evaluations**, and five randomized timing rounds for 23 configurations. Four MI100 GPUs ran independent experiments. The original MLP remained the primary control.

## Highlights

- **Rank-1 KS2, Hermite basis:** at 100k updates on the larger training pool, reserved loss was **0.005482 nats/byte lower than MLP**, winning all three seeds. It uses **73,600 learned parameters per FFN**, versus **73,728 for MLP**.
- **Combining rank-1 with a power basis:** reserved gaps were **−0.001364** on the original pool and **−0.002809** on the larger pool, with three seed wins against MLP in each. The simpler Hermite variant still led on the larger pool; gains did not accumulate universally.
- **MLP in its own subspace:** a 32-feature ReLU² branch alongside mixed-rank KS2 used **72,916 parameters per FFN**. Its 15k loss was effectively tied with its KS2-only parent: exploratory change **−0.000217**, reserved change **+0.000122**. Every mixture included two rank-16 charts per FFN.
- **Median training step:** MLP **7.47 ms**, rank-1 KS2 **8.12 ms**, mixed KS2 **9.01 ms**, mixed KS2 + MLP **9.12 ms** (B4 × 64 bytes, FP32, one MI100). Round drift and full timing ranges are documented.

These are results for a four-layer, 96-dimensional, byte-level model. They do not establish a universal KS2 advantage. The report separates parameter counts, fixed storage, geometry cost and whole-step speed, and compares this study with the prior RWKV, attention, history and RHAC work.

## Evidence

- [Full report (Markdown)](REPORT.md) · [Standalone HTML](report.html)
- [Run ledger CSV](all_runs.csv) and [JSON](all_runs.json)
- [Fresh reserved scores](final_checks/holdout_results.json)
- [Controlled timings](final_checks/speed_results.json)
- [Checkpoint/source audit](audit.json) and [protocol audit](protocol_audit.json)
- [Research completion review](goal_completion_review.json)
- [Exact source bundle](source_bundle.tar.gz) and [bundle verification](source_bundle_check.json)
- [Publication file manifest](publication_manifest.json)

The package preserves every local file linked directly by the report, plus supporting manifests and summaries. It excludes datasets, checkpoints and bulk runtime logs. Paths inside raw JSON records describe the original local experiment workspace; they are provenance, not a claim that those large artifacts are included. The source archive contains exact recorded versions for all 428 runs and separately labeled analysis tools. Its data and checkpoint hashes allow comparison with the retained local artifacts.
