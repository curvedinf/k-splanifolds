"""Train/evaluate the raw-KS2-history, pre-RWKV-mux architecture for 15k steps."""
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
    command = [sys.executable, str(root / 'run_raw_mux_gpu.py'),
               '--data', str(root / 'gpu_runs/data'), '--out', str(out),
               '--variant', 'inner_ks2_ffn', '--width', '96', '--layers', '4',
               '--ffn-charts', '64', '--seed', '11', '--steps', '15000', '--control-update-scale']
    (args.out / 'command.json').write_text(json.dumps(command, indent=2)+'\n')
    with (out / 'run.log').open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((out / 'results.json').read_text())
    assert result['parameters'] == 543136 and result['invalid_steps'] == 0
    assert result['ffn_types'] == ['ks2']*4
    assert result['arguments']['steps'] == result['history'][-1]['step'] == 15000
    for name, expected in result['source_hashes'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    with (out / 'tiled.log').open('w') as log:
        subprocess.run([sys.executable, str(root / 'evaluate_raw_mux.py'),
                        '--runs', str(args.out), '--arm', 'fixed_both', '--seeds', '11'],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    tiled = json.loads((args.out / 'tiled_fixed_both.json').read_text())[0]
    comparisons = {}
    for name,file in [('no_history','side_stream_2048_c64'), ('demuxed_history','side_history')]:
        baseline = json.loads((root / (file+'_summary.json')).read_text())
        previous = json.loads((root / (file+'_results.json')).read_text())
        assert result['data'] == previous['data']
        comparisons[name] = dict(parameters=baseline['parameters'],
            full_shard_loss=baseline['tiled_evaluation']['loss'],
            loss_delta=tiled['loss']-baseline['tiled_evaluation']['loss'],
            benchmark_ms_per_step=baseline['benchmark_ms_per_step'],
            sustained_ms_per_step=baseline['sustained_ms_per_step'])
    summary = dict(architecture=result['architecture'], charts=64, rank=8, layers=4,
        attention_width=96, attention_input_widths=[96,704,1312,1920],
        ffn_input_widths=[192,800,1408,2016], raw_output_width=512, final_demux_width=96,
        steps=15000, seed=11, parameters=result['parameters'],
        parameter_breakdown=result['parameter_breakdown'],
        benchmark_ms_per_step=result['benchmark_ms_per_step'],
        sustained_ms_per_step=result['training_seconds']/15,
        evaluation=result['evaluation'], tiled_evaluation=tiled, comparisons=comparisons,
        protocol='Same seed/data/15k-step FP32 optimizer schedule and physical GPU0. '
                 'All four KS2 FFNs emit raw 512D coordinates; history retains embedding and all prior '
                 '96D post-attention and 512D KS2 states without detach. RWKV reads a fixed Hadamard '
                 '96D view of history. Current post-attention state joins KS2 mux input. '
                 'Only final layer demuxes to the logits head, with unit-init gains at 5.5x LR. '
                 'No 2048D side stream or main-stream bypass. Initial logits differ; this is an '
                 'architecture comparison, not an isolated change with paired initial functions. '
                 'One seed; full-shard evaluation visits the same held-out bytes at four overlapping offsets.')
    (args.out / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root / 'raw_mux_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root / 'raw_mux_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
