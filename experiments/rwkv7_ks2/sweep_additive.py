"""Three paired additive-FFN runs plus a concurrent all-MLP control replay."""
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
    parser.add_argument('--initial-out', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=105000)
    args = parser.parse_args()
    if args.steps < 15000:
        parser.error('Final steps must be at least 15000')
    root = Path(__file__).parent
    jobs = [(f'additive_{seed}', 'inner_mlp_plus_ks2', seed) for seed in (11, 22, 33)]
    jobs.append(('mlp_replay_11', 'mlp', 11))
    for name, _, _ in jobs:
        if (args.out / name).exists() or (args.initial_out / name).exists():
            raise RuntimeError(f'Use fresh output directories; {name} already exists')

    def worker(item):
        gpu, (name, variant, seed) = item
        initial, final = args.initial_out / name, args.out / name
        initial.mkdir(parents=True)
        final.mkdir(parents=True)
        flags = ['--control-update-scale'] if variant == 'inner_mlp_plus_ks2' else []
        command = [sys.executable, str(root / 'run_gpu.py'), '--data', str(args.data),
                   '--out', str(initial), '--variant', variant, '--width', '96', '--layers', '4',
                   '--seed', str(seed), '--ffn-charts', '2', '--steps', '15000', *flags]
        print(json.dumps(dict(event='initial', run=name, gpu=gpu)), flush=True)
        with (initial / 'run.log').open('w') as log:
            subprocess.run(command, env=environment(gpu), stdout=log, stderr=subprocess.STDOUT, check=True)
        command = [sys.executable, str(root / 'continue_gpu.py'), '--resume', str(initial / 'checkpoint.pt'),
                   '--data', str(args.data), '--out', str(final), '--steps', str(args.steps)]
        if seed == 11:
            command.append('--verify-resume')
        print(json.dumps(dict(event='continue', run=name, gpu=gpu)), flush=True)
        with (final / 'run.log').open('w') as log:
            subprocess.run(command, env=environment(gpu), stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((final / 'results.json').read_text())
        assert result['arguments']['steps'] == args.steps
        print(json.dumps(dict(event='complete', run=name, gpu=gpu, **result['history'][-1])), flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, enumerate(jobs)))
    print('All additive runs and control replay complete', flush=True)


if __name__ == '__main__':
    main()
