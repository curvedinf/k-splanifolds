"""Validate completed short projection ablations and save reviewable artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

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
    summary = json.loads((args.runs / 'summary.json').read_text())
    initial_losses = {}
    data_hashes = set()
    labels = dict(mlp='All MLP', learned='KS2: learned projections',
                  fixed_input='KS2: Hadamard input, learned output',
                  fixed_both='KS2: Hadamard input and output')
    expected = dict(mlp=615936, learned=477904, fixed_input=474832, fixed_both=471952)
    for arm, rows in results.items():
        assert [r['arguments']['seed'] for r in rows] == [11, 22, 33]
        for row in rows:
            cfg = row['arguments']
            assert row['invalid_steps'] == 0 and row['parameters'] == expected[arm]
            assert cfg['steps'] == row['history'][-1]['step'] == 15000
            assert (cfg['width'], cfg['layers'], cfg['ffn_charts']) == (96, 4, 2)
            assert row['training_targets'] == 3840000
            initial_losses.setdefault(cfg['seed'], []).append(row['history'][0]['val_loss'])
            data_hashes.add(tuple(row['data'][split]['sha256'] for split in ('train', 'val')))
            for filename, digest in row['source_hashes'].items():
                assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == digest, filename
    assert len(data_hashes) == 1
    assert all(len(set(losses)) == 1 for losses in initial_losses.values())
    for arm in ('fixed_input', 'fixed_both'):
        differences = [r['evaluation']['expanded_val'] - c['evaluation']['expanded_val']
                       for r, c in zip(results[arm], results['learned'])]
        summary['arms'][arm]['paired_gap_vs_learned_ks2'] = dict(
            mean=float(np.mean(differences)), sd=float(np.std(differences, ddof=1)), values=differences)
    tiled_path = root / 'spread2_tiled_validation.json'
    if tiled_path.exists():
        tiled = json.loads(tiled_path.read_text())
        for arm in results:
            rows = [r for r in tiled['rows'] if r['arm'] == arm]
            assert [r['seed'] for r in rows] == [11, 22, 33]
            assert all(r['targets'] == 1777024 for r in rows)
            values = [r['loss'] for r in rows]
            summary['arms'][arm]['tiled_val'] = dict(mean=float(np.mean(values)),
                                                    sd=float(np.std(values, ddof=1)), values=values)
        for arm in ('fixed_input', 'fixed_both'):
            gaps = np.array(summary['arms'][arm]['tiled_val']['values']) - np.array(
                summary['arms']['learned']['tiled_val']['values'])
            summary['arms'][arm]['tiled_gap_vs_learned_ks2'] = dict(
                mean=float(gaps.mean()), sd=float(gaps.std(ddof=1)), values=gaps.tolist())
    summary['protocol'] = dict(
        seeds=[11, 22, 33], steps=15000, training_targets=3840000,
        width=96, layers=4, middle_ffns_only=True, charts=2, rank=8,
        code_seed=103415, all_inputs_in_each_coordinate=True,
        optimizer='AdamW, cosine base LR 0.001 to 0.0002, clip 1; KS2 output weights/gains LR x5.5 with inverse WD adjustment',
        input_code='two independent RHAC permute -> 8-way Hadamard -> signs views; sum 12 groups / sqrt(12)',
        fixed_output='normalized transpose of the 16x96 code, followed by 96 trainable zero-initialized gains',
        eval='8192 original / 65536 expanded held-out byte targets; 65536 sampled training targets',
        limitations=['one fixed coding seed across training seeds',
                     'coding changes coordinate scale and output parameterization; not optimizer-tuned',
                     'short convergence comparison, not a plateau or general LLM claim',
                     'no added depth-history input; adaptation of RHAC encoder only'])
    for name in ('summary', 'results'):
        data = summary if name == 'summary' else results
        (root / f'spread2_{name}.json').write_text(json.dumps(data, indent=2) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for arm, rows in results.items():
        steps = np.array([h['step'] for h in rows[0]['history']])
        vals = np.array([[h['val_loss'] for h in r['history']] for r in rows])
        axes[0].plot(steps[1:], vals.mean(0)[1:], label=labels[arm])
        axes[0].fill_between(steps[1:], (vals.mean(0)-vals.std(0, ddof=1))[1:],
                             (vals.mean(0)+vals.std(0, ddof=1))[1:], alpha=.12)
        gaps = vals - np.array([[h['val_loss'] for h in r['history']] for r in results['mlp']])
        axes[1].plot(steps[1:], gaps.mean(0)[1:], label=labels[arm])
    axes[0].set(ylabel='Validation loss (nats/byte)', title='Original fixed 8,192-target validation sample')
    axes[1].set(ylabel='Loss difference from all-MLP', title='Paired mean gap (lower is better)')
    axes[1].axhline(0, color='black', linewidth=.6)
    for ax in axes:
        ax.set_xlabel('Training steps')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Two rank-8 KS2 charts; all 96 inputs in each Hadamard coordinate\nThree seeds, same 15,000 steps and data stream')
    fig.tight_layout()
    fig.savefig(root / 'spread2_curves.png', dpi=160)
    fig.savefig(root / 'spread2_curves.svg')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
