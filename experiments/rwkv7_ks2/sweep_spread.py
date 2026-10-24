"""Four simultaneous MI100 arms, three paired seeds, fixed 15k-step short runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import subprocess
import sys
from sweep_plateau import environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=15000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parent
    arms = [('mlp', 'mlp', 'learned'), ('learned', 'inner_ks2_ffn', 'learned'),
            ('fixed_input', 'inner_ks2_ffn', 'fixed_input'),
            ('fixed_both', 'inner_ks2_ffn', 'fixed_both')]

    def worker(item):
        gpu, (name, variant, coding) = item
        rows = []
        for seed in args.seeds:
            out = args.out / f'{name}_{seed}'
            out.mkdir()
            command = [sys.executable, str(root / 'run_spread_gpu.py'), '--coding', coding,
                       '--data', str(root / 'gpu_runs/data'), '--out', str(out), '--variant', variant,
                       '--width', '96', '--layers', '4', '--ffn-charts', '2', '--seed', str(seed),
                       '--steps', str(args.steps)]
            if variant != 'mlp':
                command.append('--control-update-scale')
            print(json.dumps(dict(event='start', arm=name, seed=seed, gpu=gpu)), flush=True)
            with (out / 'run.log').open('w') as log:
                subprocess.run(command, env=environment(gpu), stdout=log, stderr=subprocess.STDOUT, check=True)
            row = json.loads((out / 'results.json').read_text())
            rows.append(row)
            print(json.dumps(dict(event='complete', arm=name, seed=seed, gpu=gpu,
                                  evaluation=row['evaluation'], seconds=row['training_seconds'])), flush=True)
        return name, rows

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = dict(pool.map(worker, enumerate(arms)))
    summary = {}
    for arm, rows in results.items():
        summary[arm] = dict(parameters=rows[0]['parameters'], seeds=args.seeds,
                            training_seconds=[r['training_seconds'] for r in rows])
        for metric in ('original_val', 'expanded_val', 'sampled_train'):
            values = [r['evaluation'][metric] for r in rows]
            summary[arm][metric] = dict(mean=statistics.mean(values),
                                        sd=statistics.stdev(values) if len(values) > 1 else 0., values=values)
        gaps = [r['evaluation']['expanded_val'] - m['evaluation']['expanded_val']
                for r, m in zip(rows, results['mlp'])]
        summary[arm]['paired_gap_vs_mlp'] = dict(mean=statistics.mean(gaps), values=gaps)
    (args.out / 'summary.json').write_text(json.dumps(dict(steps=args.steps, arms=summary), indent=2) + '\n')
    (args.out / 'all_results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
