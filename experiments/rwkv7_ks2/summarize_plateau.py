"""Archive and plot matched-step continuations after the 1/16 validation plateau."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent
    plateau = json.loads((args.runs / 'plateau.json').read_text())
    assert plateau['status'] == 'complete'
    steps = plateau['selected_steps']
    seeds = plateau['protocol']['seeds']
    configs = [('mlp', 'All MLP'), ('hybrid_c16', 'Middle KS2: 1/2 budget'),
               ('hybrid_c2', 'Middle KS2: 1/16 budget')]
    records, summary = {}, {}
    reference = None
    for name, label in configs:
        rows = []
        for seed in seeds:
            path = args.runs / f'{name}_{seed}'
            result = json.loads((path / 'results.json').read_text())
            checkpoint = torch.load(path / 'checkpoint.pt', map_location='cpu', weights_only=True)
            assert result == checkpoint['run_result']
            assert checkpoint['step'] == result['arguments']['steps'] == steps
            assert all(s['step'].item() == steps for s in checkpoint['optimizer']['state'].values())
            assert all(torch.isfinite(v).all() for v in checkpoint['model'].values())
            assert result['invalid_steps'] == 0
            assert result['history'][-1]['step'] == steps
            assert result['continuation']['start_step'] in (15000, 100000)
            assert result['continuation']['base_lr'] == .0002
            assert result['arguments']['width'] == 96 and result['arguments']['layers'] == 4
            assert result['ffn_types'] == (['mlp'] * 4 if name == 'mlp' else ['mlp', 'ks2', 'ks2', 'mlp'])
            if reference is None:
                reference = result
                for source, expected in result['source_hashes'].items():
                    candidates = [root / source, root / 'source_snapshots' / 'pre_additive' / source]
                    assert any(p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == expected
                               for p in candidates)
                assert hashlib.sha256((root / 'continue_gpu.py').read_bytes()).hexdigest() == result['continuation']['source_sha256']
            for key in ['data', 'source_hashes', 'batch_size', 'sequence_length', 'precision']:
                assert result[key] == reference[key]
            assert result['continuation']['source_sha256'] == reference['continuation']['source_sha256']
            # Same random windows within each seed, including every continued draw.
            if name != 'mlp':
                control = torch.load(args.runs / f'mlp_{seed}' / 'checkpoint.pt',
                                     map_location='cpu', weights_only=True)
                assert torch.equal(checkpoint['sampler_state'], control['sampler_state'])
                assert result['history'][0] == records[f'mlp_{seed}']['history'][0]
            records[path.name] = result
            rows.append(result)
        record = dict(label=label, parameters=rows[0]['parameters'],
                      per_middle_ffn_parameters=rows[0]['ffn_parameters_by_layer'][1],
                      total_steps=steps, training_targets=steps * 256,
                      training_seconds_mean=statistics.mean(r['training_seconds'] for r in rows),
                      continuation_seconds_mean=statistics.mean(
                          r['training_seconds'] - next(h['training_seconds'] for h in r['history'] if h['step'] == 15000)
                          for r in rows))
        for metric in ['monitor_loss', 'val_loss', 'sampled_train_loss']:
            start = [next(h for h in r['history'] if h['step'] == 15000)[metric] for r in rows]
            final = [r['history'][-1][metric] for r in rows]
            gaps = [v - records[f'mlp_{seed}']['history'][-1][metric] for seed, v in zip(seeds, final)]
            start_gaps = [v - next(h for h in records[f'mlp_{seed}']['history'] if h['step'] == 15000)[metric]
                          for seed, v in zip(seeds, start)]
            record[metric] = dict(start_mean=statistics.mean(start), mean=statistics.mean(final),
                                  std=statistics.stdev(final), seeds=final,
                                  improvement=statistics.mean(start) - statistics.mean(final),
                                  paired_gaps=gaps, gap_mean=statistics.mean(gaps),
                                  gap_std=statistics.stdev(gaps), start_gap_mean=statistics.mean(start_gaps))
        histories = [[h for h in r['history'] if h['step'] >= 15000] for r in rows]
        assert all([h['step'] for h in history] == [h['step'] for h in histories[0]] for history in histories)
        means = [statistics.mean(h[i]['monitor_loss'] for h in histories) for i in range(len(histories[0]))]
        best = min(range(len(means)), key=means.__getitem__)
        record['best_mean_monitor'] = dict(step=histories[0][best]['step'], loss=means[best])
        summary[name] = record
    payload = dict(plateau=plateau, models=summary,
                   evaluation_note='Monitor and training losses use fixed 65,536-byte samples. Original validation uses 8,192 targets. Same existing shards; not an independent test set.',
                   schedule='Original 15k cosine from 0.001 to 0.0002, then constant base LR 0.0002 through the selected step. Existing KS2 output-projection LR and decay scaling preserved.',
                   checks=dict(checkpoints=len(records), optimizer_steps=steps, finite_parameters=True,
                               matching_model_sources=True, matching_data=True, matching_sampler_states=True),
                   analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root / 'hybrid4_plateau_results.json').write_text(json.dumps(records, indent=2) + '\n')
    (root / 'hybrid4_plateau_summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2), flush=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    metrics = [('monitor_loss', 'Validation (65,536 byte targets)'),
               ('sampled_train_loss', 'Training sample (65,536 byte targets)'),
               ('monitor_loss', 'Validation gap versus all MLP')]
    for name, label in configs:
        histories = [[h for h in records[f'{name}_{seed}']['history'] if h['step'] >= 15000] for seed in seeds]
        x = np.array([h['step'] for h in histories[0]]) / 1000
        for index, (metric, title) in enumerate(metrics):
            values = np.array([[h[metric] for h in history] for history in histories])
            if index == 2:
                control = np.array([[h[metric] for h in records[f'mlp_{seed}']['history'] if h['step'] >= 15000]
                                    for seed in seeds])
                values = values - control
            mean, std = values.mean(0), values.std(0, ddof=1)
            line, = axes[index].plot(x, mean, label=label, linewidth=1.8)
            axes[index].fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=.12)
            axes[index].set_title(title, fontsize=11)
    for index, ax in enumerate(axes):
        ax.set_xlabel('Total training steps (thousands)')
        ax.set_ylabel('Loss (nats/byte)' if index < 2 else 'KS2 minus MLP (nats/byte)')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[2].axhline(0, color='black', linewidth=.7)
    fig.suptitle(f'Four-layer width-96 RWKV: continued from 15k to {steps:,} steps; three seeds, sample SD')
    fig.savefig(root / 'hybrid4_plateau_curves.png', dpi=180)
    fig.savefig(root / 'hybrid4_plateau_curves.svg')


if __name__ == '__main__':
    main()
