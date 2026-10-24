"""Three 32-chart Hadamard seeds plus a same-width learned-projection control."""
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
    parser.add_argument('--steps', type=int, default=15000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parent
    jobs = [(f'fixed_both_{seed}', 'fixed_both', seed) for seed in (11, 22, 33)]
    jobs.append(('learned_11', 'learned', 11))

    def worker(item):
        gpu, (name, coding, seed) = item
        out = args.out / name
        out.mkdir()
        command = [sys.executable, str(root / 'run_spread32_gpu.py'), '--coding', coding,
                   '--data', str(root / 'gpu_runs/data'), '--out', str(out),
                   '--variant', 'inner_ks2_ffn', '--width', '96', '--layers', '4',
                   '--ffn-charts', '32', '--seed', str(seed), '--steps', str(args.steps),
                   '--control-update-scale']
        print(json.dumps(dict(event='start', run=name, gpu=gpu)), flush=True)
        with (out / 'run.log').open('w') as log:
            subprocess.run(command, env=environment(gpu), stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((out / 'results.json').read_text())
        print(json.dumps(dict(event='complete', run=name, gpu=gpu, evaluation=result['evaluation'],
                              parameters=result['parameters'], seconds=result['training_seconds'])), flush=True)
        return name, result

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = dict(pool.map(worker, enumerate(jobs)))
    (args.out / 'all_results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
