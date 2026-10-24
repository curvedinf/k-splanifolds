"""After the completed side sweep, benchmark references and train one 2048-side run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from sweep_plateau import environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    completed = json.loads((args.sweep / 'all_results.json').read_text())
    assert set(completed) == {'0','32','96','160'}
    assert all(len(rows) == 3 and all(r['arguments']['steps'] == 15000 for r in rows)
               for rows in completed.values()), 'Finish the current sweep first'
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parent
    env = environment(0)
    benchmarks = {}
    for side, steps in ((0,0), (160,0), (2048,15000)):
        out = args.out / (f'benchmark_side{side}' if not steps else 'fixed_both_11')
        out.mkdir()
        command = [sys.executable, str(root / 'run_side_gpu.py'), '--side-width', str(side),
                   '--data', str(root / 'gpu_runs/data'), '--out', str(out),
                   '--variant', 'inner_ks2_ffn', '--width', '96', '--layers', '4',
                   '--ffn-charts', '32', '--seed', '11', '--steps', str(steps), '--control-update-scale']
        print(json.dumps(dict(event='start', side=side, steps=steps, gpu=0)), flush=True)
        with (out / 'run.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        record = json.loads((out / ('results.json' if steps else 'benchmark.json')).read_text())
        benchmarks[side] = record['benchmark_ms_per_step']
        print(json.dumps(dict(event='complete', side=side, benchmark_ms_per_step=benchmarks[side])), flush=True)
    result = json.loads((args.out / 'fixed_both_11/results.json').read_text())
    with (args.out / 'fixed_both_11/tiled.log').open('w') as log:
        subprocess.run([sys.executable, str(root / 'evaluate_side.py'), '--side-width', '2048',
                        '--runs', str(args.out), '--arm', 'fixed_both', '--seeds', '11'],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    tiled = json.loads((args.out / 'tiled_fixed_both.json').read_text())[0]
    summary = dict(main_width=96, side_width=2048, ffn_width=2144, steps=15000, seed=11,
                   parameters=result['parameters'], parameter_breakdown=result['parameter_breakdown'],
                   benchmark_ms_same_gpu=benchmarks,
                   benchmark_slowdown_vs_no_side=benchmarks[2048] / benchmarks[0] - 1,
                   benchmark_slowdown_vs_side160=benchmarks[2048] / benchmarks[160] - 1,
                   sustained_ms_per_step=result['training_seconds'] / 15,
                   evaluation=result['evaluation'], tiled_evaluation=tiled,
                   protocol='Same physical GPU0, sequential fresh-process 100-update benchmarks; '
                            'warmup and benchmark updates reset. One full 15k-step seed-11 run for 2048 side channels. '
                            'Benchmark steps include forward, loss, backward, clipping, AdamW, batch copies and LR writes; exclude setup/eval. '
                            'Full-shard evaluation reuses the same shard at four overlapping offsets.')
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (root / 'side_stream_2048_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (root / 'side_stream_2048_results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
