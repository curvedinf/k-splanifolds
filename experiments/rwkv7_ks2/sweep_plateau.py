"""Continue 1/16-budget seeds to a validation plateau, then matched controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from queue import Empty, Queue
import statistics
import subprocess
import sys
import time


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def environment(gpu):
    env = dict(os.environ, ROCR_VISIBLE_DEVICES=str(gpu), ROCM_PATH='/opt/rocm',
               LD_LIBRARY_PATH='/opt/rocm/lib', PYTORCH_ROCM_ARCH='gfx908', GPU_ARCHS='gfx908')
    env.pop('HIP_VISIBLE_DEVICES', None)
    env.pop('CUDA_VISIBLE_DEVICES', None)
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path(__file__).parent / 'gpu_runs')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=5000)
    parser.add_argument('--patience', type=int, default=20000)
    parser.add_argument('--min-improvement', type=float, default=.002)
    parser.add_argument('--max-steps', type=int,
                        help='Optional hard cap, checked before dispatching another training chunk')
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 15000:
        parser.error('The cap must be at least the starting checkpoint step (15000)')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'plateau.json').exists():
        raise RuntimeError('Use a fresh output directory to preserve existing experiments')
    seeds = [11, 22, 33]
    command_file = args.out / 'command.json'
    target = 15000
    atomic_json(command_file, dict(target_step=target, stop=False))
    worker = Path(__file__).with_name('continue_gpu.py')
    data = args.runs / 'data'
    processes, logs = [], []
    record = dict(protocol=dict(seeds=seeds, interval=args.interval, patience=args.patience,
                                min_improvement=args.min_improvement, base_lr=.0002,
                                monitor_targets=65536, start_step=target, max_steps=args.max_steps),
                  history=[], status='sixteenth')
    try:
        for gpu, seed in enumerate(seeds):
            dest = args.out / f'hybrid_c2_{seed}'
            dest.mkdir(parents=True, exist_ok=True)
            log = (dest / 'run.log').open('w')
            logs.append(log)
            resume = args.runs / 'hybrid4_tiny_budgets' / dest.name / 'checkpoint.pt'
            processes.append(subprocess.Popen([sys.executable, str(worker), '--resume', str(resume),
                                                '--data', str(data), '--out', str(dest),
                                                '--command', str(command_file)],
                                               env=environment(gpu), stdout=log, stderr=subprocess.STDOUT))
        while True:
            results = []
            for seed, proc in zip(seeds, processes):
                if proc.poll() is not None:
                    raise RuntimeError(f'Sixteenth worker seed {seed} exited with {proc.returncode}')
                path = args.out / f'hybrid_c2_{seed}' / 'results.json'
                if path.exists():
                    result = json.loads(path.read_text())
                    if result['arguments']['steps'] == target:
                        results.append(result)
            if len(results) != len(seeds):
                time.sleep(1)
                continue
            entry = dict(step=target)
            for metric in ['monitor_loss', 'val_loss', 'sampled_train_loss']:
                values = [r['history'][-1][metric] for r in results]
                entry[metric] = statistics.mean(values)
                entry[metric + '_seeds'] = values
            record['history'].append(entry)
            older = [p['monitor_loss'] for p in record['history'] if p['step'] <= target - args.patience]
            best = min(p['monitor_loss'] for p in record['history'])
            improvement = min(older) - best if older else None
            entry['best_improvement_over_patience'] = improvement
            print(json.dumps(entry), flush=True)
            plateau_reached = improvement is not None and improvement < args.min_improvement
            cap_reached = args.max_steps is not None and target >= args.max_steps
            if plateau_reached or cap_reached:
                record.update(status='controls', selected_steps=target,
                              stopping_reason='Validation plateau criterion met' if plateau_reached else 'Step cap reached')
                atomic_json(args.out / 'plateau.json', record)
                atomic_json(command_file, dict(target_step=target, stop=True))
                break
            atomic_json(args.out / 'plateau.json', record)
            target += args.interval
            if args.max_steps is not None:
                target = min(target, args.max_steps)
            atomic_json(command_file, dict(target_step=target, stop=False))
        for proc in processes:
            if proc.wait(timeout=30) != 0:
                raise RuntimeError('Sixteenth worker failed during shutdown')
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=30)
        for log in logs:
            log.close()

    print(f'Stopped at {target} steps; starting all-MLP and half-budget controls', flush=True)
    jobs = Queue()
    for seed in seeds:
        for name, source_name in [('mlp', 'mlp'), ('hybrid_c16', 'hybrid')]:
            jobs.put((name, source_name, seed))

    def controls(gpu):
        while True:
            try:
                name, source_name, seed = jobs.get_nowait()
            except Empty:
                return
            dest = args.out / f'{name}_{seed}'
            dest.mkdir(parents=True, exist_ok=True)
            resume = args.runs / 'hybrid4' / f'{source_name}_{seed}' / 'checkpoint.pt'
            with (dest / 'run.log').open('w') as log:
                subprocess.run([sys.executable, str(worker), '--resume', str(resume), '--data', str(data),
                                '--out', str(dest), '--steps', str(target)], env=environment(gpu),
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((dest / 'results.json').read_text())
            print(json.dumps(dict(run=dest.name, **result['history'][-1])), flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(controls, range(4)))
    record['status'] = 'complete'
    atomic_json(args.out / 'plateau.json', record)
    print('All nine matched-step runs complete', flush=True)


if __name__ == '__main__':
    main()
