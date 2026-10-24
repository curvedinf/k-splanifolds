"""Four-layer paired screen, or an inner-KS2 FFN budget sweep."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=15000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33])
    parser.add_argument('--ffn-charts', type=int, nargs='+',
                        help='Run only these KS2 widths; omit for MLP versus 16-subspace KS2')
    args = parser.parse_args()
    jobs = Queue()
    configs = ([(f'hybrid_c{charts}', charts) for charts in args.ffn_charts]
               if args.ffn_charts else [('mlp', None), ('hybrid', 16)])
    for seed in args.seeds:
        for name, charts in configs:
            jobs.put((name, seed, charts))

    def worker(gpu):
        env = dict(os.environ, ROCR_VISIBLE_DEVICES=str(gpu), ROCM_PATH='/opt/rocm',
                   LD_LIBRARY_PATH='/opt/rocm/lib', PYTORCH_ROCM_ARCH='gfx908', GPU_ARCHS='gfx908')
        env.pop('HIP_VISIBLE_DEVICES', None)
        env.pop('CUDA_VISIBLE_DEVICES', None)
        while True:
            try:
                name, seed, charts = jobs.get_nowait()
            except Empty:
                return
            dest = args.out / f'{name}_{seed}'
            dest.mkdir(parents=True, exist_ok=True)
            flags = ['--variant', 'mlp'] if charts is None else [
                '--variant', 'inner_ks2_ffn', '--ffn-charts', str(charts), '--control-update-scale']
            command = [sys.executable, str(Path(__file__).with_name('run_gpu.py')),
                       '--data', str(args.data), '--out', str(dest), '--width', '96',
                       '--layers', '4', '--steps', str(args.steps), '--seed', str(seed), *flags]
            with (dest / 'run.log').open('w') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((dest / 'results.json').read_text())
            print(json.dumps(dict(config=name, seed=seed,
                                  final_loss=result['history'][-1]['val_loss'],
                                  training_seconds=result['training_seconds'])), flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, range(4)))


if __name__ == '__main__':
    main()
