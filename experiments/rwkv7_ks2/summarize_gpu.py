"""Archive paired GPU results and plot training-duration comparisons."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--short', type=Path, required=True)
    parser.add_argument('--long', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent
    records = {}
    for regime, directory in [('short', args.short), ('long', args.long)]:
        records[regime] = {p.parent.name: json.loads(p.read_text())
                           for p in sorted(directory.glob('*/results.json'))}
    (root / 'gpu_training_results.json').write_text(json.dumps(records, indent=2) + '\n')
    configs = ['mlp', 'ks2', 'ffn11', 'ffn22_scaled']
    seeds = [11, 22, 33]
    summary = {}
    for regime, rows in records.items():
        summary[regime] = {}
        for config in configs:
            runs = [rows[f'{config}_{seed}'] for seed in seeds]
            losses = [r['history'][-1]['val_loss'] for r in runs]
            differences = [loss - rows[f'mlp_{seed}']['history'][-1]['val_loss']
                           for loss, seed in zip(losses, seeds)]
            summary[regime][config] = dict(
                mean_loss=statistics.mean(losses), std_loss=statistics.stdev(losses),
                paired_gap_mean=statistics.mean(differences),
                paired_gap_std=statistics.stdev(differences),
                mean_training_seconds=statistics.mean(r['training_seconds'] for r in runs),
                mean_wall_seconds=statistics.mean(r['wall_seconds'] for r in runs),
                losses=losses, paired_gaps=differences)
    (root / 'gpu_training_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    labels = ['Original MLP', 'KS2 controllers', 'KS2 FFNs, 11 subspaces',
              'KS2 FFNs, 22 subspaces + scaled LR']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    rows = records['long']
    for config, label in zip(configs, labels):
        histories = [rows[f'{config}_{seed}']['history'][1:] for seed in seeds]
        steps = np.array([h['step'] for h in histories[0]])
        values = np.array([[h['val_loss'] for h in history] for history in histories])
        mean, std = values.mean(0), values.std(0, ddof=1)
        line, = axes[0].plot(steps, mean, label=label)
        axes[0].fill_between(steps, mean - std, mean + std, color=line.get_color(), alpha=.15)
        if config != 'mlp':
            base = np.array([[h['val_loss'] for h in rows[f'mlp_{seed}']['history'][1:]]
                             for seed in seeds])
            gaps = values - base
            mean, std = gaps.mean(0), gaps.std(0, ddof=1)
            axes[1].plot(steps, mean, label=label, color=line.get_color())
            axes[1].fill_between(steps, mean - std, mean + std, color=line.get_color(), alpha=.15)
    axes[0].set_ylabel('Validation cross-entropy (nats/byte)')
    axes[0].set_title('All models improve with longer training')
    axes[1].axhline(0, color='black', lw=.8)
    axes[1].set_ylabel('Paired loss difference from original MLP')
    axes[1].set_title('Does the KS2 gap close?')
    for ax in axes:
        ax.set_xlabel('Training steps')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('RWKV-7 on MI100: 15,000 steps, three paired seeds; shading = sample SD')
    fig.savefig(root / 'gpu_training_curves.png', dpi=180)
    fig.savefig(root / 'gpu_training_curves.svg')


if __name__ == '__main__':
    main()
