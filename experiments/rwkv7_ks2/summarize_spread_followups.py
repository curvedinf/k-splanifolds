"""Validate and summarize 32-chart and wider-residual Hadamard follow-ups."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent


def stats(values):
    return dict(mean=statistics.mean(values),
                sd=statistics.stdev(values) if len(values) > 1 else None, values=values)


def check(row, expected_parameters):
    cfg = row['arguments']
    assert cfg['steps'] == row['history'][-1]['step'] == 15000
    assert cfg['ffn_charts'] == 32 and cfg['layers'] == 4
    assert row['parameters'] == expected_parameters and row['invalid_steps'] == 0
    assert row['training_targets'] == 3840000
    assert row['batch_size'] == 4 and row['sequence_length'] == 64
    assert abs(row['evaluation']['original_val'] - row['history'][-1]['val_loss']) < 2e-6
    for name, digest in row['source_hashes'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    tiled = row['tiled_evaluation']
    assert tiled['targets'] == 1777024
    assert tiled['seed'] == cfg['seed']
    assert tiled['batch4_vs_batch64_max_loss_error'] < 2e-6


def aggregate(rows):
    rows = sorted(rows, key=lambda r: r['arguments']['seed'])
    result = dict(parameters=rows[0]['parameters'],
                  inner_ffn_parameters=rows[0]['ffn_parameters_by_layer'][1],
                  seeds=[r['arguments']['seed'] for r in rows],
                  training_seconds=[r['training_seconds'] for r in rows],
                  output_lr_multiplier=rows[0]['output_lr_multiplier'])
    for metric in ('original_val', 'expanded_val', 'sampled_train'):
        result[metric] = stats([r['evaluation'][metric] for r in rows])
    result['tiled_val'] = stats([r['tiled_evaluation']['loss'] for r in rows])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-widths', action='store_true')
    args = parser.parse_args()
    source = ROOT / 'gpu_runs/spread32_short'
    records = json.loads((source / 'all_results.json').read_text())
    for name, row in records.items():
        row['tiled_evaluation'] = json.loads((source / f'tiled_{name}.json').read_text())[0]
        check(row, 521152 if row['coding'] == 'fixed_both' else 619264)
    had = [records[f'fixed_both_{seed}'] for seed in (11, 22, 33)]
    learned = [records['learned_11']]
    earlier = json.loads((ROOT / 'spread2_summary.json').read_text())
    summary = dict(steps=15000, width=96, charts=32, rank=8,
                   fixed_both=aggregate(had), learned_control_seed11=aggregate(learned),
                   previous_two_chart_fixed=earlier['arms']['fixed_both'],
                   previous_all_mlp=earlier['arms']['mlp'],
                   protocol='Same 15k-step/3.84M-target protocol. Fixed gains retain LR multiplier 5.5 '
                            'and unit-norm decoder rows; dense output control uses 11/32. '
                            '32 fixed independent codes, all inputs in every coordinate. '
                            'Learned 32-chart control has only seed 11; other means use 11/22/33.')
    (ROOT / 'spread32_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (ROOT / 'spread32_results.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if not args.include_widths:
        return
    widths = json.loads((ROOT / 'gpu_runs/spread_widths_short/all_results.json').read_text())
    counts = {128: 808448, 192: 1579648, 256: 2612992}
    for row in widths:
        check(row, counts[row['arguments']['width']])
        assert row['data'] == had[0]['data']
        assert row['coding'] == 'fixed_both' and row['output_lr_multiplier'] == 5.5
        # The reused 96-wide runner's output description is static text;
        # actual gain tensor size always follows the configured model width.
        width = row['arguments']['width']
        row['coding_protocol']['output'] = (
            f'fixed transpose with unit row norms and {width} zero-init learned output gains')
    groups = {96: had}
    for width in counts:
        groups[width] = [r for r in widths if r['arguments']['width'] == width]
        assert sorted(r['arguments']['seed'] for r in groups[width]) == [11, 22, 33]
    width_summary = dict(steps=15000, layers=4, charts=32, rank=8,
                        widths={width: aggregate(rows) for width, rows in groups.items()},
                        protocol='Three seeds per width, 3.84M training targets/run; same data, optimizer, '
                                 'base LR and fixed-gain LR, fixed 32 rank-8 inner KS2 charts. '
                                 'Each subspace receives a distinct fixed Hadamard mixture of every input. '
                                 'Embedding, recurrence, outer 4x MLPs and logits head grow with width. '
                                 'Same validation shard, four overlapping offsets; not an independent test set.',
                        limitation='Width-matched MLP controls have only seed 11; three-seed Hadamard means '
                                   'should not be interpreted as a paired comparison against those single runs.')
    for width, group in width_summary['widths'].items():
        group['paired_tiled_gap_vs_width96'] = stats([
            a-b for a,b in zip(group['tiled_val']['values'], width_summary['widths'][96]['tiled_val']['values'])])
    control_counts = {128: 1017856, 192: 2116608, 256: 3608576}
    controls = {}
    for width, count in control_counts.items():
        control_path = ROOT / f'gpu_runs/spread_width_mlp_controls/width{width}/combined.json'
        if not control_path.exists():
            continue
        row = json.loads(control_path.read_text())
        check(row, count)
        assert row['arguments']['variant'] == 'mlp' and row['arguments']['seed'] == 11
        controls[width] = row
    if controls:
        width_summary['width_matched_mlp_seed11'] = {w: aggregate([r]) for w,r in controls.items()}
        width_summary['seed11_hadamard_minus_mlp'] = {
            w: next(r['tiled_evaluation']['loss'] for r in groups[w] if r['arguments']['seed'] == 11)
               - controls[w]['tiled_evaluation']['loss'] for w in controls}
        (ROOT / 'spread_widths_mlp_controls.json').write_text(json.dumps(controls, indent=2) + '\n')
    (ROOT / 'spread_widths_summary.json').write_text(json.dumps(width_summary, indent=2) + '\n')
    (ROOT / 'spread_widths_results.json').write_text(json.dumps(widths, indent=2) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for width, rows in groups.items():
        steps = [h['step'] for h in rows[0]['history']][1:]
        vals = np.array([[h['val_loss'] for h in r['history']][1:] for r in rows])
        axes[0].plot(steps, vals.mean(0), label=f'Width {width}')
        axes[0].fill_between(steps, vals.mean(0)-vals.std(0,ddof=1), vals.mean(0)+vals.std(0,ddof=1), alpha=.12)
    xs = list(groups)
    means = [width_summary['widths'][w]['tiled_val']['mean'] for w in xs]
    errors = [width_summary['widths'][w]['tiled_val']['sd'] for w in xs]
    axes[1].errorbar(xs, means, yerr=errors, marker='o', capsize=4)
    if controls:
        axes[1].plot(list(controls), [r['tiled_evaluation']['loss'] for r in controls.values()],
                     marker='s', linestyle='--', label='Width-matched MLP (seed 11 only)')
    axes[1].axhline(earlier['arms']['mlp']['tiled_val']['mean'], linestyle='--', color='gray', label='Width-96 MLP reference')
    axes[0].set(xlabel='Training steps', ylabel='Validation loss (nats/byte)', title='Fixed 8,192-target validation sample')
    axes[1].set(xlabel='Residual width', ylabel='Full-shard loss (nats/byte)', title='Final loss: mean ± seed SD', xticks=xs)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('32 rank-8 Hadamard KS2 subspaces; 15,000 steps; three seeds per width')
    fig.tight_layout()
    fig.savefig(ROOT / 'spread_widths_curves.png', dpi=160)
    fig.savefig(ROOT / 'spread_widths_curves.svg')
    print(json.dumps(width_summary, indent=2))


if __name__ == '__main__':
    main()
