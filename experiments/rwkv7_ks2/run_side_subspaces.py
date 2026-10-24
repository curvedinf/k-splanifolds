"""Train/evaluate 64 rank-8 subspaces with the existing 2048-channel side stream."""
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
    command = [sys.executable, str(root / 'run_side_gpu.py'), '--side-width', '2048',
               '--data', str(root / 'gpu_runs/data'), '--out', str(out),
               '--variant', 'inner_ks2_ffn', '--width', '96', '--layers', '4',
               '--ffn-charts', '64', '--seed', '11', '--steps', '15000', '--control-update-scale']
    (args.out / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (out / 'run.log').open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((out / 'results.json').read_text())
    assert result['parameters'] == 564096 and result['invalid_steps'] == 0
    assert result['arguments']['ffn_charts'] == result['coding_protocol']['charts'] == 64
    assert result['arguments']['steps'] == result['history'][-1]['step'] == 15000
    for name, expected in result['source_hashes'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    with (out / 'tiled.log').open('w') as log:
        subprocess.run([sys.executable, str(root / 'evaluate_side.py'), '--side-width', '2048',
                        '--runs', str(args.out), '--arm', 'fixed_both', '--seeds', '11'],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    tiled = json.loads((args.out / 'tiled_fixed_both.json').read_text())[0]
    baseline = json.loads((root / 'side_stream_2048_summary.json').read_text())
    summary = dict(main_width=96, side_width=2048, ffn_width=2144, charts=64, rank=8,
                   steps=15000, seed=11, parameters=result['parameters'],
                   parameter_breakdown=result['parameter_breakdown'],
                   benchmark_ms_per_step=result['benchmark_ms_per_step'],
                   sustained_ms_per_step=result['training_seconds'] / 15,
                   evaluation=result['evaluation'], tiled_evaluation=tiled,
                   comparison_32charts=dict(
                       parameters=baseline['parameters'],
                       full_shard_loss=baseline['tiled_evaluation']['loss'],
                       full_shard_loss_delta=tiled['loss']-baseline['tiled_evaluation']['loss'],
                       benchmark_ms_per_step=baseline['benchmark_ms_same_gpu']['2048'],
                       sustained_ms_per_step=baseline['sustained_ms_per_step']),
                   protocol='Same seed 11, 15k steps, data, optimizer schedule and output-gain LR 5.5x; '
                            '64 instead of 32 rank-8 subspaces in all four KS2 FFNs. '
                            'Same physical GPU0 as earlier 32-chart run. One seed per configuration. '
                            'Fixed coding and geometry initialization change with subspace count; '
                            'shared main-stream initialization and initial logits are preserved. '
                            'Full-shard evaluation uses four overlapping offsets on the same held-out shard.')
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (root / 'side_stream_2048_c64_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (root / 'side_stream_2048_c64_results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
