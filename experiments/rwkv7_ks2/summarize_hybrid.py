"""Archive the four-layer hybrid comparison, including expanded evaluation."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent
    rows = {p.parent.name: json.loads(p.read_text()) for p in args.runs.glob('*/results.json')}
    assert len(rows) == 6
    seeds = [11, 22, 33]
    summary = {}
    evaluation = json.loads(args.evaluation.read_text())['rows']
    for name in ('mlp', 'hybrid'):
        runs = [rows[f'{name}_{seed}'] for seed in seeds]
        losses = [r['history'][-1]['val_loss'] for r in runs]
        extra = [r for r in evaluation if r['run'].startswith(name + '_')]
        summary[name] = dict(parameters=runs[0]['parameters'],
                             ffn_parameters_by_layer=runs[0]['ffn_parameters_by_layer'],
                             mean_loss=statistics.mean(losses), std_loss=statistics.stdev(losses),
                             losses=losses,
                             training_seconds=statistics.mean(r['training_seconds'] for r in runs),
                             wall_seconds=statistics.mean(r['wall_seconds'] for r in runs),
                             expanded_val=statistics.mean(r['expanded_val'] for r in extra),
                             sampled_train=statistics.mean(r['sampled_train'] for r in extra))
    gaps = [rows[f'hybrid_{seed}']['history'][-1]['val_loss'] -
            rows[f'mlp_{seed}']['history'][-1]['val_loss'] for seed in seeds]
    summary['paired_gap'] = dict(mean=statistics.mean(gaps), std=statistics.stdev(gaps), values=gaps)
    for name, value in [('hybrid4_results.json', rows), ('hybrid4_summary.json', summary)]:
        (root / name).write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    arrays = {}
    for name, label in [('mlp', 'MLP / MLP / MLP / MLP'), ('hybrid', 'MLP / KS2 / KS2 / MLP')]:
        histories = [rows[f'{name}_{seed}']['history'][1:] for seed in seeds]
        steps = [h['step'] for h in histories[0]]
        values = np.array([[h['val_loss'] for h in history] for history in histories])
        arrays[name] = values
        mean, std = values.mean(0), values.std(0, ddof=1)
        line, = axes[0].plot(steps, mean, label=label)
        axes[0].fill_between(steps, mean - std, mean + std, color=line.get_color(), alpha=.15)
    difference = arrays['hybrid'] - arrays['mlp']
    for seed, values in zip(seeds, difference):
        axes[1].plot(steps, values, alpha=.5, lw=1, label=f'Seed {seed}')
    axes[1].plot(steps, difference.mean(0), color='black', lw=2, label='Paired mean')
    axes[1].axhline(0, color='gray', lw=.8)
    axes[0].set_ylabel('Validation loss (nats/byte)')
    axes[1].set_ylabel('Hybrid minus all-MLP loss (nats/byte)')
    for ax in axes:
        ax.set_xlabel('Training steps')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Four-layer RWKV, width 96: half-budget KS2 only in the middle FFNs')
    fig.savefig(root / 'hybrid4_curves.png', dpi=180)
    fig.savefig(root / 'hybrid4_curves.svg')

    a, b = summary['mlp'], summary['hybrid']
    lines = ['# Four smaller RWKV layers with KS2 only in the middle FFNs', '',
             f"The hybrid's mean final validation loss is **{b['mean_loss']:.6f}**, versus **{a['mean_loss']:.6f}** for the all-MLP control: a paired difference of **{statistics.mean(gaps):+.6f} nats/byte**. Total parameters decrease **{100 * (1 - b['parameters'] / a['parameters']):.2f}%**.", '',
             'Both models have four RWKV layers at width 96, with three heads of size 32. Each layer contains time mixing and an FFN, with separate residual connections. All memory controllers use the original parameterization in both arms. Only FFNs in layers 2 and 3 change.', '',
             '| Model | Layer 1 FFN | Layer 2 FFN | Layer 3 FFN | Layer 4 FFN | Total parameters |',
             '|---|---|---|---|---|---:|',
             f"| Control | MLP | MLP | MLP | MLP | {a['parameters']:,} |",
             f"| Hybrid | MLP | KS2 | KS2 | MLP | {b['parameters']:,} |", '',
             'Each ordinary FFN is 96 → 384 → 96 with ReLU-squared activation and 73,728 parameters. Each middle KS2 FFN has 16 rank-8 subspaces, one geometry step, and 37,696 parameters: **51.13%** of the replaced MLP, the closest whole-subspace choice to half its budget. Outer FFNs retain the full 73,728 parameters each. Both models train their 256 × 96 embedding and separate, untied 96 → 256 logits head.', '',
             '## Final results', '',
             'Three paired seeds (11, 22, 33), 15,000 steps from scratch. Loss is nats/byte; ± denotes sample standard deviation. Final checkpoints are reported.', '',
             '| Model | Validation loss | Training seconds/run | Process wall seconds/run |',
             '|---|---:|---:|---:|',
             f"| All MLP | {a['mean_loss']:.6f} ± {a['std_loss']:.6f} | {a['training_seconds']:.1f} | {a['wall_seconds']:.1f} |",
             f"| Middle KS2 | {b['mean_loss']:.6f} ± {b['std_loss']:.6f} | {b['training_seconds']:.1f} | {b['wall_seconds']:.1f} |", '',
             '| Seed | All MLP | Middle KS2 | KS2 minus MLP |', '|---|---:|---:|---:|']
    for seed, gap in zip(seeds, gaps):
        lines.append(f"| {seed} | {rows[f'mlp_{seed}']['history'][-1]['val_loss']:.6f} | {rows[f'hybrid_{seed}']['history'][-1]['val_loss']:.6f} | {gap:+.6f} |")
    lines += ['', '![Training curves and paired differences](hybrid4_curves.png)', '',
              '## Additional evaluation', '',
              'Each saved checkpoint was evaluated on the same extra 65,536 held-out byte targets and 65,536 sampled training targets used in the preceding GPU follow-up. The held-out shard is unchanged and windows may overlap; this is not an independent test set.', '',
              '| Model | Expanded validation | Sampled training |', '|---|---:|---:|',
              f"| All MLP | {a['expanded_val']:.6f} | {a['sampled_train']:.6f} |",
              f"| Middle KS2 | {b['expanded_val']:.6f} | {b['sampled_train']:.6f} |", '',
              '## Controls and verification', '',
              '- Same prepared FineWeb bytes, batches within each seed, 32 original validation batches, batch size 4, sequence length 64, FP32 arithmetic, and zero recurrent state per window as the preceding GPU runs. Each run sees 3,840,000 training byte targets.',
              '- AdamW, gradient clipping at norm 1, and cosine LR from approximately 0.001 to 0.0002 over 15,000 steps. The established width correction is applied only to the two KS2 FFN output projections: LR multiplier 11/16, with inverse weight-decay scaling to preserve multiplicative shrinkage. No hyperparameter sweep was performed.',
              '- All retained parameters are identically initialized within each paired seed. Zero output projections give identical initial logits. `check_hybrid.py` verifies shared parameters, exact placement, counts, optimizer grouping, initial logits, and full-model GPU forward/gradient parity against the original recurrence.',
              '- All runs reject nonfinite loss/gradient measurements. Checkpoint optimizer counters equal 15,000 and loaded checkpoints reproduce the logged final validation losses.',
              '- Timing is measured per independent GPU run; one process per MI100, not tensor parallelism. Training excludes startup, graph calibration, validation, and checkpoint writes. The GPU backend and tested int8-vllm environment are described in [the preceding report](GPU_TRAINING_REPORT.md).',
              '- Width 96 is smaller than the earlier width 128. Four layers change total depth and parameter counts, so the causal comparison is the paired four-layer control, not the old two-layer model. Three seeds and short sequences do not establish a general architecture ranking.', '',
              '## Reproduce', '', '```bash',
              'ROCR_VISIBLE_DEVICES=0 ROCM_PATH=/opt/rocm LD_LIBRARY_PATH=/opt/rocm/lib PYTORCH_ROCM_ARCH=gfx908 GPU_ARCHS=gfx908 ../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/check_hybrid.py',
              '../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/sweep_hybrid.py --data experiments/rwkv7_ks2/gpu_runs/data --out experiments/rwkv7_ks2/gpu_runs/hybrid4 --steps 15000',
              'ROCR_VISIBLE_DEVICES=0 ROCM_PATH=/opt/rocm LD_LIBRARY_PATH=/opt/rocm/lib PYTORCH_ROCM_ARCH=gfx908 GPU_ARCHS=gfx908 ../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/analyze_gpu.py --data experiments/rwkv7_ks2/gpu_runs/data --runs experiments/rwkv7_ks2/gpu_runs/hybrid4 --out experiments/rwkv7_ks2/hybrid4_expanded_evaluation.json',
              '../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/summarize_hybrid.py --runs experiments/rwkv7_ks2/gpu_runs/hybrid4 --evaluation experiments/rwkv7_ks2/hybrid4_expanded_evaluation.json',
              '```', '',
              'Raw configurations, curves, source hashes, and timings are in `hybrid4_results.json`; aggregates are in `hybrid4_summary.json`. Checkpoints and logs remain under ignored `gpu_runs/hybrid4/`.', '']
    (root / 'HYBRID4_REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
