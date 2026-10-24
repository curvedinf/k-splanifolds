"""Compare middle KS2 FFN budgets against the four-layer MLP."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--additional-runs', type=Path)
    parser.add_argument('--additional-evaluation', type=Path)
    args = parser.parse_args()
    if bool(args.additional_runs) != bool(args.additional_evaluation):
        parser.error('Additional runs and evaluation must be supplied together')
    root = Path(__file__).parent
    records = json.loads((root / 'hybrid4_results.json').read_text())
    records.update({p.parent.name: json.loads(p.read_text()) for p in args.runs.glob('*/results.json')})
    extra = json.loads((root / 'hybrid4_expanded_evaluation.json').read_text())['rows']
    extra += json.loads(args.evaluation.read_text())['rows']
    if args.additional_runs:
        records.update({p.parent.name: json.loads(p.read_text())
                        for p in args.additional_runs.glob('*/results.json')})
        extra += json.loads(args.additional_evaluation.read_text())['rows']
    extra = {r['run']: r for r in extra}
    configs = [('mlp', 'All MLP'), ('hybrid_c31', 'Middle KS2, full budget'),
               ('hybrid', 'Middle KS2, half budget'), ('hybrid_c8', 'Middle KS2, quarter budget')]
    if args.additional_runs:
        configs += [('hybrid_c4', 'Middle KS2, eighth budget'),
                    ('hybrid_c2', 'Middle KS2, sixteenth budget')]
    seeds = [11, 22, 33]
    summary = {}
    for name, label in configs:
        runs = [records[f'{name}_{seed}'] for seed in seeds]
        losses = [r['history'][-1]['val_loss'] for r in runs]
        gaps = [loss - records[f'mlp_{seed}']['history'][-1]['val_loss']
                for seed, loss in zip(seeds, losses)]
        for seed, run in zip(seeds, runs):
            control = records[f'mlp_{seed}']
            assert run['history'][-1]['step'] == 15000 and run['invalid_steps'] == 0
            assert run['data'] == control['data']
            assert run['history'][0]['val_loss'] == control['history'][0]['val_loss']
            assert run['source_hashes'] == control['source_hashes']
            assert run['arguments']['width'] == 96 and run['arguments']['layers'] == 4
            assert run['ffn_types'] == (['mlp'] * 4 if name == 'mlp' else ['mlp', 'ks2', 'ks2', 'mlp'])
        summary[name] = dict(label=label, parameters=runs[0]['parameters'],
                             per_middle_ffn_parameters=runs[0]['ffn_parameters_by_layer'][1],
                             middle_ffn_fraction=runs[0]['ffn_parameters_by_layer'][1] / 73728,
                             mean_loss=statistics.mean(losses), std_loss=statistics.stdev(losses),
                             losses=losses, paired_gaps=gaps,
                             gap_mean=statistics.mean(gaps), gap_std=statistics.stdev(gaps),
                             training_seconds=statistics.mean(r['training_seconds'] for r in runs),
                             wall_seconds=statistics.mean(r['wall_seconds'] for r in runs),
                             expanded_val=statistics.mean(extra[f'{name}_{seed}']['expanded_val'] for seed in seeds),
                             sampled_train=statistics.mean(extra[f'{name}_{seed}']['sampled_train'] for seed in seeds))
    (root / 'hybrid4_budget_results.json').write_text(json.dumps(records, indent=2) + '\n')
    (root / 'hybrid4_budget_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for name, label in configs:
        histories = [records[f'{name}_{seed}']['history'][1:] for seed in seeds]
        steps = [h['step'] for h in histories[0]]
        values = np.array([[h['val_loss'] for h in history] for history in histories])
        mean, std = values.mean(0), values.std(0, ddof=1)
        line, = axes[0].plot(steps, mean, label=label)
        axes[0].fill_between(steps, mean - std, mean + std, color=line.get_color(), alpha=.12)
    budget_rows = sorted([summary[k] for k, _ in configs if k != 'mlp'],
                         key=lambda r: r['middle_ffn_fraction'])
    axes[1].errorbar([r['middle_ffn_fraction'] * 100 for r in budget_rows],
                     [r['mean_loss'] for r in budget_rows],
                     yerr=[r['std_loss'] for r in budget_rows], marker='o', capsize=4,
                     color='tab:orange', label='Middle KS2 FFNs')
    control = summary['mlp']
    axes[1].axhline(control['mean_loss'], color='tab:blue', label='All-MLP control')
    axes[1].axhspan(control['mean_loss'] - control['std_loss'],
                    control['mean_loss'] + control['std_loss'], color='tab:blue', alpha=.12)
    axes[0].set_xlabel('Training steps')
    axes[1].set_xlabel('Parameters per middle FFN (% of MLP)')
    if args.additional_runs:
        ticks = [r['middle_ffn_fraction'] * 100 for r in budget_rows]
        axes[1].set_xscale('log', base=2)
        axes[1].set_xticks(ticks, labels=[f'{v:.1f}' for v in ticks])
    for ax in axes:
        ax.set_ylabel('Validation loss (nats/byte)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Four-layer RWKV, width 96: middle FFN budgets; three paired seeds, sample SD')
    fig.savefig(root / 'hybrid4_budget_curves.png', dpi=180)
    fig.savefig(root / 'hybrid4_budget_curves.svg')

    full, quarter = summary['hybrid_c31'], summary['hybrid_c8']
    new_count = 12 if args.additional_runs else 6
    tested = 'full, quarter, eighth, and sixteenth' if args.additional_runs else 'full and quarter'
    lines = ['# Four-layer RWKV: middle KS2 FFN parameter budgets', '',
             f"Full-budget middle KS2 FFNs reach **{full['mean_loss']:.6f}** mean validation loss, and quarter-budget middle KS2 FFNs reach **{quarter['mean_loss']:.6f}**, versus **{control['mean_loss']:.6f}** for all MLP. Their paired gaps are **{full['gap_mean']:+.6f}** and **{quarter['gap_mean']:+.6f} nats/byte**, respectively.", '',
             f'Follow-up to the [half-budget hybrid](HYBRID4_REPORT.md). {new_count} new runs test approximately {tested} MLP parameter budgets in the two middle KS2 FFNs. The all-MLP and half-budget controls are reused unchanged. Every configuration uses four width-96 layers, original memory controllers, full MLP FFNs in layers 1 and 4, and trained untied byte embeddings and logits heads.', '',
             'A budget refers to each replaced middle FFN. Whole rank-8 subspaces give 31 subspaces (73,036 parameters, 99.06% of MLP), 8 (18,848 parameters, 25.56%), 4 (9,424 parameters, 12.78%), and 2 (4,712 parameters, 6.39%). The previous half-budget model uses 16 subspaces (37,696 parameters, 51.13%). Each full MLP FFN has 73,728 parameters. No unused parameters are added to make counts exact.', '',
             '## Results', '',
             'Final loss after 15,000 steps, averaged over paired seeds 11, 22, and 33. Loss is cross-entropy in nats/byte; ± is sample standard deviation.', '',
             '| Configuration | Parameters per middle FFN | Total parameters | Validation loss | Paired gap vs MLP | Training seconds/run |',
             '|---|---:|---:|---:|---:|---:|']
    if args.additional_runs:
        eighth, sixteenth = summary['hybrid_c4'], summary['hybrid_c2']
        lines[4:4] = [
            f"The eighth-budget and sixteenth-budget extensions reach **{eighth['mean_loss']:.6f}** and **{sixteenth['mean_loss']:.6f}**, with paired gaps of **{eighth['gap_mean']:+.6f}** and **{sixteenth['gap_mean']:+.6f} nats/byte**.", '']
    for name, label in configs:
        r = summary[name]
        lines.append(f"| {label} | {r['per_middle_ffn_parameters']:,} | {r['parameters']:,} | {r['mean_loss']:.6f} ± {r['std_loss']:.6f} | {r['gap_mean']:+.6f} | {r['training_seconds']:.1f} |")
    headings = ['MLP', 'Full KS2', 'Half KS2', 'Quarter KS2']
    if args.additional_runs:
        headings += ['Eighth KS2', 'Sixteenth KS2']
    lines += ['', '| Seed | ' + ' | '.join(headings) + ' |', '|---|' + '---:|' * len(headings)]
    for i, seed in enumerate(seeds):
        lines.append('| ' + str(seed) + ' | ' + ' | '.join(f"{summary[name]['losses'][i]:.6f}" for name, _ in configs) + ' |')
    lines += ['', '![Training curves and parameter-budget comparison](hybrid4_budget_curves.png)', '',
              '## Expanded evaluation', '',
              'The same additional 65,536 byte targets from the held-out shard and 65,536 sampled training targets are used for all configurations. These are additional fixed windows from existing shards, not an independent test set; windows may overlap.', '',
              '| Configuration | Expanded validation | Gap vs MLP | Sampled training |', '|---|---:|---:|---:|']
    for name, label in configs:
        r = summary[name]
        lines.append(f"| {label} | {r['expanded_val']:.6f} | {r['expanded_val'] - control['expanded_val']:+.6f} | {r['sampled_train']:.6f} |")
    if args.additional_runs:
        lines += ['',
                  'The original-window means worsen gradually as the middle FFNs shrink, without a sharp collapse. Full-budget KS2 still trails all MLP in every seed. The eighth/sixteenth ranking reverses on the larger validation sample, so these three-seed results do not establish a reliable ordering between the two smallest budgets. Because only two FFNs are compressed, the sixteenth-budget model saves 22.41% of total model parameters rather than reducing the entire model to one sixteenth.', '']
    lines += ['', '## Protocol and limits', '',
              '- All new runs start from scratch, use FP32, batch 4, sequence length 64, 15,000 updates, and 3,840,000 training byte targets. Data, paired minibatches, initialization procedure, final-checkpoint selection, and the cosine schedule match the preceding four-layer experiment.',
              '- The same width-aware optimizer rule is retained: only KS2 FFN output-projection LR is multiplied by 11/subspaces (11/31, 11/8, 11/4, or 11/2). Its weight-decay coefficient is divided by that multiplier to preserve AdamW shrinkage. All other parameters use the original LR schedule. This controls the known first-update width scaling approximately; it does not enforce identical optimization throughout training.',
              '- Existing all-MLP and half-budget results are reused. The archive verifies identical runner/model source hashes, data manifests, initial paired validation losses, and architecture. Parameter counts and shared initialization were checked before the runs. Loaded final checkpoints reproduce their logged validation losses and optimizer counters equal 15,000.',
              '- Each run uses one MI100; four processes can execute concurrently. Training seconds exclude startup, warmup, calibration, validation, and checkpoint writes. Tested software and runtime hashes are in `gpu_environment.json`.',
              '- This is a small, short-context, three-seed screen. The budget variants are compared on the same validation windows; selecting a winner does not constitute independent confirmation or establish an optimal budget.', '',
              '## Reproduce', '', '```bash',
              '../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/sweep_hybrid.py --data experiments/rwkv7_ks2/gpu_runs/data --out experiments/rwkv7_ks2/gpu_runs/hybrid4_budgets --steps 15000 --ffn-charts 31 8',
              'ROCR_VISIBLE_DEVICES=0 ROCM_PATH=/opt/rocm LD_LIBRARY_PATH=/opt/rocm/lib PYTORCH_ROCM_ARCH=gfx908 GPU_ARCHS=gfx908 ../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/analyze_gpu.py --data experiments/rwkv7_ks2/gpu_runs/data --runs experiments/rwkv7_ks2/gpu_runs/hybrid4_budgets --out experiments/rwkv7_ks2/hybrid4_budget_expanded_evaluation.json',
              '../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/summarize_hybrid_budgets.py --runs experiments/rwkv7_ks2/gpu_runs/hybrid4_budgets --evaluation experiments/rwkv7_ks2/hybrid4_budget_expanded_evaluation.json',
              '```', '',
              'The preceding half-budget report describes how to regenerate the reused controls. Combined raw results: `hybrid4_budget_results.json`; aggregate statistics: `hybrid4_budget_summary.json`. New checkpoints and logs remain under ignored `gpu_runs/hybrid4_budgets/`.', '']
    if args.additional_runs:
        lines += ['To reproduce the eighth/sixteenth extension and combined report:', '', '```bash',
                  '../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/sweep_hybrid.py --data experiments/rwkv7_ks2/gpu_runs/data --out experiments/rwkv7_ks2/gpu_runs/hybrid4_tiny_budgets --steps 15000 --ffn-charts 4 2',
                  'ROCR_VISIBLE_DEVICES=0 ROCM_PATH=/opt/rocm LD_LIBRARY_PATH=/opt/rocm/lib PYTORCH_ROCM_ARCH=gfx908 GPU_ARCHS=gfx908 ../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/analyze_gpu.py --data experiments/rwkv7_ks2/gpu_runs/data --runs experiments/rwkv7_ks2/gpu_runs/hybrid4_tiny_budgets --out experiments/rwkv7_ks2/hybrid4_tiny_expanded_evaluation.json',
                  '../int8-vllm/.venv/bin/python experiments/rwkv7_ks2/summarize_hybrid_budgets.py --runs experiments/rwkv7_ks2/gpu_runs/hybrid4_budgets --evaluation experiments/rwkv7_ks2/hybrid4_budget_expanded_evaluation.json --additional-runs experiments/rwkv7_ks2/gpu_runs/hybrid4_tiny_budgets --additional-evaluation experiments/rwkv7_ks2/hybrid4_tiny_expanded_evaluation.json',
                  '```', '',
                  'The additional checkpoints and logs remain under ignored `gpu_runs/hybrid4_tiny_budgets/`.', '']
    (root / 'HYBRID4_BUDGET_REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
