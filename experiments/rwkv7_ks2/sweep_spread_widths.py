"""Queue nine width/seed jobs across four MI100s, including full-shard evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from sweep_plateau import environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=15000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parent
    queue = Queue()
    for width in (128, 192, 256):
        for seed in (11, 22, 33):
            queue.put((width, seed))

    def worker(gpu):
        completed = []
        while True:
            try:
                width, seed = queue.get_nowait()
            except Empty:
                return completed
            group = args.out / f'width{width}'
            out = group / f'fixed_both_{seed}'
            out.mkdir(parents=True)
            command = [sys.executable, str(root / 'run_spread_width_gpu.py'), '--coding', 'fixed_both',
                       '--data', str(root / 'gpu_runs/data'), '--out', str(out),
                       '--variant', 'inner_ks2_ffn', '--width', str(width), '--layers', '4',
                       '--ffn-charts', '32', '--seed', str(seed), '--steps', str(args.steps),
                       '--control-update-scale']
            print(json.dumps(dict(event='start', width=width, seed=seed, gpu=gpu)), flush=True)
            with (out / 'run.log').open('w') as log:
                subprocess.run(command, env=environment(gpu), stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((out / 'results.json').read_text())
            print(json.dumps(dict(event='trained', width=width, seed=seed, gpu=gpu,
                                  evaluation=result['evaluation'], seconds=result['training_seconds'])), flush=True)
            tiled = group / f'tiled_{seed}.json'
            with (out / 'tiled.log').open('w') as log:
                subprocess.run([sys.executable, str(root / 'evaluate_spread.py'), '--runs', str(group),
                                '--arm', 'fixed_both', '--seeds', str(seed), '--output', str(tiled)],
                               env=environment(gpu), stdout=log, stderr=subprocess.STDOUT, check=True)
            result['tiled_evaluation'] = json.loads(tiled.read_text())[0]
            completed.append(result)
            print(json.dumps(dict(event='complete', width=width, seed=seed, gpu=gpu,
                                  tiled_val=result['tiled_evaluation']['loss'])), flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = [row for rows in pool.map(worker, range(4)) for row in rows]
    results.sort(key=lambda r: (r['arguments']['width'], r['arguments']['seed']))
    (args.out / 'all_results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
