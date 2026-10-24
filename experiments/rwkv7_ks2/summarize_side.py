"""Validate and summarize all-KS2 FFNs with a side-only residual stream."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent
    results = json.loads((args.runs / 'all_results.json').read_text())
    initial, data = {}, set()
    groups = {}
    for side, rows in results.items():
        side = int(side)
        rows.sort(key=lambda r:r['arguments']['seed'])
        assert [r['arguments']['seed'] for r in rows] == [11, 22, 33]
        for row in rows:
            cfg = row['arguments']
            assert (cfg['width'], cfg['layers'], cfg['ffn_charts'], cfg['side_width']) == (96, 4, 32, side)
            assert row['parameters'] == 426368 + 16 * side
            assert row['ffn_types'] == ['ks2'] * 4
            assert row['ffn_parameters_by_layer'] == [26240 + 96 + side] * 4
            assert row['invalid_steps'] == 0
            assert cfg['steps'] == row['history'][-1]['step'] == 15000
            assert row['training_targets'] == 3840000 and row['output_lr_multiplier'] == 5.5
            assert row['parameter_breakdown']['rwkv'] == 269568
            assert row['parameter_breakdown']['embedding_and_head'] == 49152
            assert row['parameter_breakdown']['ks2_geometry'] == 104960
            assert row['tiled_evaluation']['targets'] == 1777024
            assert row['tiled_evaluation']['batch4_vs_batch64_max_loss_error'] < 2e-6
            assert abs(row['evaluation']['original_val'] - row['history'][-1]['val_loss']) < 2e-6
            initial.setdefault(cfg['seed'], []).append(row['history'][0]['val_loss'])
            data.add(tuple(row['data'][split]['sha256'] for split in ('train', 'val')))
            for name,digest in row['source_hashes'].items():
                assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
        group = dict(parameters=rows[0]['parameters'], parameter_breakdown=rows[0]['parameter_breakdown'],
                     main_width=96, side_width=side, ffn_width=96+side, seeds=[11,22,33])
        metrics = {name:[r['evaluation'][name] for r in rows]
                   for name in ('original_val', 'expanded_val', 'sampled_train')}
        metrics.update(tiled_val=[r['tiled_evaluation']['loss'] for r in rows],
                       benchmark_ms=[r['benchmark_ms_per_step'] for r in rows],
                       sustained_ms=[r['training_seconds'] / 15 for r in rows])
        for name,values in metrics.items():
            group[name] = dict(mean=statistics.mean(values), sd=statistics.stdev(values), values=values)
        groups[side] = group
    assert all(len(set(values)) == 1 for values in initial.values()) and len(data) == 1
    for side,group in groups.items():
        gaps = [a-b for a,b in zip(group['tiled_val']['values'], groups[0]['tiled_val']['values'])]
        group['paired_tiled_gap_vs_no_side'] = dict(mean=statistics.mean(gaps),
                                                  sd=statistics.stdev(gaps), values=gaps)
    summary = dict(protocol=dict(
        architecture='four 32-rank8 KS2 FFNs; main 96D; side bypasses attention, embedding and head',
        side_widths=[0,32,96,160], seeds=[11,22,33], steps=15000, targets=3840000,
        side_initialization='zeros; fully differentiable across depth; final side discarded',
        ffn_input='LayerNorm over concatenated main and side, then original causal FFN time mixing',
        initialization='shared attention, main mix, embedding/head and KS2 geometry exactly paired',
        projection='32 independent fixed permute/8-way Hadamard/sign/fold codes; every FFN input in every chart coordinate',
        decoder='fixed transpose with unit row norms and zero-initialized per-output gains',
        optimizer='AdamW cosine 0.001 -> 0.0002, clip 1; gain LR x5.5 with inverse WD adjustment',
        terminal_side_gains='Final FFN side output is discarded; its side gain entries have zero task gradient and remain counted.',
        evaluation='Same held-out byte shard at four overlapping 64-token context offsets; 1,777,024 target visits/model, not independent test samples.'),
        arms=groups)
    (root / 'side_stream_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (root / 'side_stream_results.json').write_text(json.dumps(results, indent=2)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for side in (0,32,96,160):
        rows = results[str(side)]
        steps = [h['step'] for h in rows[0]['history']][1:]
        vals = np.array([[h['val_loss'] for h in r['history']][1:] for r in rows])
        axes[0].plot(steps, vals.mean(0), label=f'{side} side channels')
        axes[0].fill_between(steps, vals.mean(0)-vals.std(0,ddof=1), vals.mean(0)+vals.std(0,ddof=1), alpha=.12)
    xs = [0,32,96,160]
    axes[1].errorbar(xs, [groups[s]['tiled_val']['mean'] for s in xs],
                     yerr=[groups[s]['tiled_val']['sd'] for s in xs], marker='o', capsize=4)
    axes[0].set(xlabel='Training steps', ylabel='Validation loss (nats/byte)', title='Original 8,192-target validation sample')
    axes[1].set(xlabel='Additional side channels', ylabel='Full-shard loss (nats/byte)', title='15,000 steps: mean ± seed SD', xticks=xs)
    axes[0].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle('Four KS2 FFNs; fixed 96D main stream; three seeds per side width')
    fig.tight_layout()
    fig.savefig(root / 'side_stream_curves.png', dpi=160)
    fig.savefig(root / 'side_stream_curves.svg')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
