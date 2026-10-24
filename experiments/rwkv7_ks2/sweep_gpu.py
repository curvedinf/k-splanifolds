"""Run four matched configurations concurrently, one process per MI100."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=15000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33])
    args = parser.parse_args()
    configs = [('mlp', ['--variant', 'mlp']),
               ('ks2', ['--variant', 'ks2']),
               ('ffn11', ['--variant', 'ks2_ffn']),
               ('ffn22_scaled', ['--variant', 'ks2_ffn', '--ffn-charts', '22',
                                 '--control-update-scale'])]

    def worker(item):
        gpu, (name, flags) = item
        env = dict(os.environ, ROCR_VISIBLE_DEVICES=str(gpu), ROCM_PATH='/opt/rocm',
                   LD_LIBRARY_PATH='/opt/rocm/lib', PYTORCH_ROCM_ARCH='gfx908', GPU_ARCHS='gfx908')
        env.pop('HIP_VISIBLE_DEVICES', None)
        env.pop('CUDA_VISIBLE_DEVICES', None)
        for seed in args.seeds:
            dest = args.out / f'{name}_{seed}'
            dest.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(Path(__file__).with_name('run_gpu.py')),
                       '--data', str(args.data), '--out', str(dest),
                       '--steps', str(args.steps), '--seed', str(seed), *flags]
            with (dest / 'run.log').open('w') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((dest / 'results.json').read_text())
            print(json.dumps(dict(config=name, seed=seed,
                                  final_loss=result['history'][-1]['val_loss'],
                                  training_seconds=result['training_seconds'])), flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, enumerate(configs)))


if __name__ == '__main__':
    main()
