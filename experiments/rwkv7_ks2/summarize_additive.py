"""Compare additive middle KS2 branches with the matched 105k controls."""
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
    controls = root / 'gpu_runs' / 'hybrid4_plateau'
    seeds = (11, 22, 33)
    configurations = [('mlp', 'All MLP', controls),
                      ('hybrid_c16', 'Middle KS2: 1/2 budget', controls),
                      ('hybrid_c2', 'Middle KS2: 1/16 budget', controls),
                      ('additive', 'Full MLP + 1/16 KS2 in middle FFNs', args.runs)]
    records, summary = {}, {}
    baseline = json.loads((controls / 'mlp_11' / 'results.json').read_text())
    for name, label, folder in configurations:
        rows = []
        for seed in seeds:
            path = folder / f'{name}_{seed}'
            r = json.loads((path / 'results.json').read_text())
            checkpoint = torch.load(path / 'checkpoint.pt', map_location='cpu', weights_only=True)
            assert r == checkpoint['run_result']
            assert checkpoint['step'] == r['arguments']['steps'] == 105000
            assert all(s['step'].item() == 105000 for s in checkpoint['optimizer']['state'].values())
            assert all(torch.isfinite(p).all() for p in checkpoint['model'].values())
            assert r['invalid_steps'] == 0 and r['data'] == baseline['data']
            assert r['continuation']['base_lr'] == .0002
            if name == 'additive':
                assert r['parameters'] == 625360
                assert r['ffn_types'] == ['mlp', 'mlp+ks2', 'mlp+ks2', 'mlp']
                assert r['ffn_parameters_by_layer'] == [73728, 78440, 78440, 73728]
                assert r['history'][0] == records[f'mlp_{seed}']['history'][0]
                old = torch.load(controls / f'mlp_{seed}' / 'checkpoint.pt', map_location='cpu', weights_only=True)
                assert torch.equal(checkpoint['sampler_state'], old['sampler_state'])
                for source, expected in r['source_hashes'].items():
                    assert hashlib.sha256((root / source).read_bytes()).hexdigest() == expected
                assert hashlib.sha256((root / 'continue_gpu.py').read_bytes()).hexdigest() == r['continuation']['source_sha256']
            records[path.name] = r
            rows.append(r)
        item = dict(label=label, parameters=rows[0]['parameters'],
                    training_seconds_mean=statistics.mean(r['training_seconds'] for r in rows))
        for metric in ['monitor_loss', 'val_loss', 'sampled_train_loss']:
            losses = [r['history'][-1][metric] for r in rows]
            start = [next(h for h in r['history'] if h['step'] == 15000)[metric] for r in rows]
            gaps = [loss - records[f'mlp_{seed}']['history'][-1][metric] for seed, loss in zip(seeds, losses)]
            item[metric] = dict(mean=statistics.mean(losses), std=statistics.stdev(losses), seeds=losses,
                                start_mean=statistics.mean(start), paired_gaps=gaps,
                                gap_mean=statistics.mean(gaps), gap_std=statistics.stdev(gaps))
        summary[name] = item

    tiled = json.loads((root / 'additive4_tiled_validation.json').read_text())
    tiled_rows = {r['run']: r for r in tiled['rows']}
    for name in ('mlp', 'additive'):
        values = [tiled_rows[f'{name}_{seed}']['loss'] for seed in seeds]
        gaps = [v - tiled_rows[f'mlp_{seed}']['loss'] for seed, v in zip(seeds, values)]
        summary[name]['tiled_validation'] = dict(mean=statistics.mean(values), std=statistics.stdev(values),
                                                 seeds=values, paired_gaps=gaps,
                                                 gap_mean=statistics.mean(gaps), gap_std=statistics.stdev(gaps))

    replay_path = args.runs / 'mlp_replay_11'
    replay = json.loads((replay_path / 'results.json').read_text())
    assert replay['arguments']['steps'] == 105000 and replay['invalid_steps'] == 0
    original = torch.load(controls / 'mlp_11' / 'checkpoint.pt', map_location='cpu', weights_only=True)
    repeated = torch.load(replay_path / 'checkpoint.pt', map_location='cpu', weights_only=True)
    assert replay == repeated['run_result']
    assert repeated['step'] == 105000
    assert all(s['step'].item() == 105000 for s in repeated['optimizer']['state'].values())
    assert replay['data'] == baseline['data']
    assert replay['history'][0] == baseline['history'][0]
    assert torch.equal(repeated['sampler_state'], original['sampler_state'])
    differences = []
    for key, p in original['model'].items():
        q = repeated['model'][key]
        assert torch.isfinite(q).all()
        differences.append((p - q).abs().max().item())
    by_step = {h['step']: h for h in baseline['history']}
    metric_diff = []
    for h in replay['history']:
        for metric in ['val_loss', 'monitor_loss', 'sampled_train_loss']:
            if metric in h:
                delta = abs(h[metric] - by_step[h['step']][metric])
                metric_diff.append(delta)
    records['mlp_replay_11'] = replay
    evaluation = json.loads((root / 'additive4_evaluation.json').read_text())
    assert {r['run'] for r in evaluation['rows']} == {'additive_11', 'additive_22', 'additive_33', 'mlp_replay_11'}
    disabled, disabled_deltas = [], []
    for row in evaluation['rows']:
        last = records[row['run']]['history'][-1]
        for a, b in [('original_val', 'val_loss'), ('expanded_val', 'monitor_loss'),
                     ('sampled_train', 'sampled_train_loss')]:
            assert abs(row[a] - last[b]) < 2e-6
        if row['run'].startswith('additive_'):
            loss = row['added_ks2_disabled_at_evaluation']['expanded_val']
            disabled.append(loss)
            disabled_deltas.append(loss - row['expanded_val'])
    payload = dict(steps=105000, seeds=seeds, models=summary,
                   architecture='Layers 1 and 4: full MLP. Layers 2 and 3: full MLP(h) + 2-subspace rank-8 KS2(h), sharing normalized, time-mixed input; no branch averaging or gating.',
                   schedule='First 15k steps cosine 0.001 to 0.0002; next 90k at base LR 0.0002. KS2 output-projection LR multiplier 5.5; inverse weight-decay scaling. All MLP parameters retain original LR.',
                   evaluation='Same fixed 65,536-byte validation and training samples, plus original 8,192-byte validation. Existing shards; not an independent test set.',
                   tiled_validation_protocol=tiled['protocol'],
                   comparison_note='Primary means use three paired seeds and reuse the historical controls. The extra MLP seed-11 repeat is a separate repeatability check. No parameter-matched wider-MLP control was trained.',
                   added_ks2_disabled_at_evaluation=dict(mean_validation_loss=statistics.mean(disabled),
                                                          std_validation_loss=statistics.stdev(disabled),
                                                          mean_loss_increase=statistics.mean(disabled_deltas),
                                                          note='Zero only the added KS2 outputs after training; this is an inference-only intervention on co-adapted weights, not retraining without KS2.'),
                   mlp_replay=dict(max_parameter_difference=max(differences), max_metric_difference=max(metric_diff),
                                   final_metrics=replay['history'][-1],
                                   tiled_validation_loss=tiled_rows['mlp_replay_11']['loss'],
                                   tiled_validation_difference=tiled_rows['mlp_replay_11']['loss'] - tiled_rows['mlp_11']['loss'],
                                   final_differences={k: replay['history'][-1][k] - baseline['history'][-1][k]
                                                      for k in ['val_loss', 'monitor_loss', 'sampled_train_loss']},
                                   note='Same initialization, data, and sampler; not bit-reproducible from scratch. Repeated archived-model runs also diverged. See mlp_reproducibility.json.'),
                   checks='All final optimizer counters are 105k, all parameters finite, initial logits and sampler streams matched; additive GPU gradients and resumed Adam graph updates checked against references.',
                   analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root / 'additive4_results.json').write_text(json.dumps(records, indent=2) + '\n')
    (root / 'additive4_summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2), flush=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    selected = [('mlp', 'All MLP'), ('hybrid_c2', 'Replace middle MLPs with 1/16 KS2'),
                ('additive', 'Full MLP + 1/16 KS2 in middle layers')]
    metrics = ['monitor_loss', 'sampled_train_loss', 'monitor_loss']
    for name, label in selected:
        histories = [[h for h in records[f'{name}_{seed}']['history'] if h['step'] >= 15000] for seed in seeds]
        x = np.array([h['step'] for h in histories[0]]) / 1000
        for i, metric in enumerate(metrics):
            values = np.array([[h[metric] for h in history] for history in histories])
            if i == 2:
                values -= np.array([[h[metric] for h in records[f'mlp_{seed}']['history'] if h['step'] >= 15000]
                                    for seed in seeds])
            mean, std = values.mean(0), values.std(0, ddof=1)
            line, = axes[i].plot(x, mean, label=label, linewidth=1.8)
            axes[i].fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=.12)
    for i, title in enumerate(['Validation (65,536 byte targets)', 'Training sample (65,536 byte targets)', 'Validation gap versus all MLP']):
        axes[i].set_title(title, fontsize=11)
        axes[i].set_xlabel('Total training steps (thousands)')
        axes[i].set_ylabel('Loss (nats/byte)' if i < 2 else 'Variant minus MLP (nats/byte)')
        axes[i].grid(alpha=.2)
    axes[0].legend(fontsize=7)
    axes[2].axhline(0, color='black', linewidth=.7)
    fig.suptitle('Four-layer width-96 RWKV: additive KS2 versus replacement; three paired seeds, sample SD')
    fig.savefig(root / 'additive4_curves.png', dpi=180)
    fig.savefig(root / 'additive4_curves.svg')


if __name__ == '__main__':
    main()
