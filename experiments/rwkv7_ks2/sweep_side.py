"""Four side widths (including zero), three seeds, one independent job per GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from sweep_plateau import environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parent

    def worker(item):
        gpu, side_width = item
        rows = []
        group = args.out / f'side{side_width}'
        for seed in (11, 22, 33):
            out = group / f'fixed_both_{seed}'
            out.mkdir(parents=True)
            command = [sys.executable, str(root / 'run_side_gpu.py'), '--side-width', str(side_width),
                       '--data', str(root / 'gpu_runs/data'), '--out', str(out),
                       '--variant', 'inner_ks2_ffn', '--width', '96', '--layers', '4',
                       '--ffn-charts', '32', '--seed', str(seed), '--steps', '15000', '--control-update-scale']
            print(json.dumps(dict(event='start', side=side_width, seed=seed, gpu=gpu)), flush=True)
            with (out / 'run.log').open('w') as log:
                subprocess.run(command, env=environment(gpu), stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((out / 'results.json').read_text())
            print(json.dumps(dict(event='trained', side=side_width, seed=seed, gpu=gpu,
                                  evaluation=result['evaluation'], seconds=result['training_seconds'])), flush=True)
            tiled = group / f'tiled_{seed}.json'
            with (out / 'tiled.log').open('w') as log:
                subprocess.run([sys.executable, str(root / 'evaluate_side.py'), '--side-width', str(side_width),
                                '--runs', str(group), '--arm', 'fixed_both', '--seeds', str(seed),
                                '--output', str(tiled)], env=environment(gpu), stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            result['tiled_evaluation'] = json.loads(tiled.read_text())[0]
            rows.append(result)
            print(json.dumps(dict(event='complete', side=side_width, seed=seed, gpu=gpu,
                                  tiled_val=result['tiled_evaluation']['loss'])), flush=True)
        return side_width, rows

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = dict(pool.map(worker, enumerate((0, 32, 96, 160))))
    (args.out / 'all_results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
