"""Train and evaluate the 64-subspace/2048-side attention+FFN history ablation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from sweep_plateau import environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent
    args.out.mkdir(parents=True, exist_ok=False)
    out = args.out / 'fixed_both_11'
    out.mkdir()
    env = environment(0)
    command = [sys.executable, str(root / 'run_side_history_gpu.py'), '--side-width', '2048',
               '--data', str(root / 'gpu_runs/data'), '--out', str(out),
               '--variant', 'inner_ks2_ffn', '--width', '96', '--layers', '4',
               '--ffn-charts', '64', '--seed', '11', '--steps', '15000', '--control-update-scale']
    (args.out / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (out / 'run.log').open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((out / 'results.json').read_text())
    assert result['parameters'] == 567552 and result['invalid_steps'] == 0
    assert result['ffn_types'] == ['ks2'] * 4
    assert result['arguments']['ffn_charts'] == result['coding_protocol']['charts'] == 64
    assert result['arguments']['steps'] == result['history'][-1]['step'] == 15000
    assert result['coding_protocol']['input_widths'] == [2144, 2336, 2528, 2720]
    for name, expected in result['source_hashes'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    with (out / 'tiled.log').open('w') as log:
        subprocess.run([sys.executable, str(root / 'evaluate_side_history.py'), '--side-width', '2048',
                        '--runs', str(args.out), '--arm', 'fixed_both', '--seeds', '11'],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    tiled = json.loads((args.out / 'tiled_fixed_both.json').read_text())[0]
    baseline = json.loads((root / 'side_stream_2048_c64_summary.json').read_text())
    prior_result = json.loads((root / 'side_stream_2048_c64_results.json').read_text())
    assert result['history'][0]['val_loss'] == prior_result['history'][0]['val_loss']
    assert result['data'] == prior_result['data']
    summary = dict(main_width=96, side_width=2048, input_widths=[2144,2336,2528,2720],
                   output_width=2144, charts=64, rank=8, steps=15000, seed=11,
                   parameters=result['parameters'], parameter_breakdown=result['parameter_breakdown'],
                   benchmark_ms_per_step=result['benchmark_ms_per_step'],
                   sustained_ms_per_step=result['training_seconds'] / 15,
                   evaluation=result['evaluation'], tiled_evaluation=tiled,
                   comparison_no_history=dict(parameters=baseline['parameters'],
                       full_shard_loss=baseline['tiled_evaluation']['loss'],
                       full_shard_loss_delta=tiled['loss']-baseline['tiled_evaluation']['loss'],
                       benchmark_ms_per_step=baseline['benchmark_ms_per_step'],
                       sustained_ms_per_step=baseline['sustained_ms_per_step']),
                   protocol='Seed 11, 15k steps, same data, optimizer schedule and output-gain LR 5.5x. '
                            'History contains separate 96D post-attention and post-FFN residuals from every '
                            'strictly previous layer; all gradients retained. Expanded FFN LayerNorm/time mix. '
                            'Baseline output matrices and geometry initialization preserved; input codes change '
                            'with width. Same physical GPU0. One seed. Full-shard evaluation reuses the held-out '
                            'shard at four overlapping offsets.')
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (root / 'side_history_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (root / 'side_history_results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
