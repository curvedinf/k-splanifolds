"""Finish the user-selected 105k study, reusing all existing trained states."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
from queue import Empty, Queue
import statistics
import subprocess
import sys
import time

from sweep_plateau import atomic_json, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'gpu_runs' / 'hybrid4_plateau')
    parser.add_argument('--existing-mlp-pid', type=int,
                        help='GPU-3 process already training MLP seed 11 to 100k; otherwise launch it from 15k')
    args = parser.parse_args()
    root = Path(__file__).parent
    runs = root / 'gpu_runs'
    out = args.out
    target = 105000
    for seed in (11, 22, 33):
        result = json.loads((out / f'hybrid_c2_{seed}' / 'results.json').read_text())
        assert result['arguments']['steps'] == target
    jobs = Queue()
    for name, source in [('mlp_22', 'mlp_22'), ('hybrid_c16_11', 'hybrid_11'),
                         ('hybrid_c16_22', 'hybrid_22'), ('hybrid_c16_33', 'hybrid_33'),
                         ('mlp_33', 'mlp_33')]:
        jobs.put((name, runs / 'hybrid4' / source / 'checkpoint.pt'))

    def train(gpu, name, resume, verify=False):
        dest = out / name
        dest.mkdir(exist_ok=True)
        if (dest / 'results.json').exists():
            result = json.loads((dest / 'results.json').read_text())
            if result['arguments']['steps'] == target:
                print(json.dumps(dict(event='reuse', run=name, gpu=gpu)), flush=True)
                return
            raise RuntimeError(f'{dest} contains an incomplete run; explicitly resume its checkpoint')
        command = [sys.executable, str(root / 'continue_gpu.py'), '--resume', str(resume),
                   '--data', str(runs / 'data'), '--out', str(dest), '--steps', str(target)]
        if verify:
            command.append('--verify-resume')
        print(json.dumps(dict(event='start', run=name, gpu=gpu)), flush=True)
        with (dest / 'run.log').open('w') as log:
            subprocess.run(command, env=environment(gpu), stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        result = json.loads((dest / 'results.json').read_text())
        assert result['arguments']['steps'] == target
        print(json.dumps(dict(event='complete', run=name, gpu=gpu,
                              **result['history'][-1])), flush=True)

    def worker(gpu):
        if gpu == 3:
            if args.existing_mlp_pid:
                while Path(f'/proc/{args.existing_mlp_pid}').exists():
                    time.sleep(1)
                result = json.loads((out / 'mlp_11' / 'results.json').read_text())
                assert result['arguments']['steps'] == 100000
                archive = runs / 'hybrid4_plateau_intermediate'
                archive.mkdir(exist_ok=True)
                saved = archive / 'mlp_11_100k'
                (out / 'mlp_11').rename(saved)
                train(gpu, 'mlp_11', saved / 'checkpoint.pt', verify=True)
            else:
                train(gpu, 'mlp_11', runs / 'hybrid4' / 'mlp_11' / 'checkpoint.pt')
        while True:
            try:
                name, resume = jobs.get_nowait()
            except Empty:
                return
            train(gpu, name, resume)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, range(4)))
    record = json.loads((out / 'plateau.json').read_text())
    rows = [json.loads((out / f'hybrid_c2_{seed}' / 'results.json').read_text()) for seed in (11, 22, 33)]
    histories = [[h for h in r['history'] if h['step'] >= 15000] for r in rows]
    record['history'] = []
    for index in range(len(histories[0])):
        entry = dict(step=histories[0][index]['step'])
        for metric in ['monitor_loss', 'val_loss', 'sampled_train_loss']:
            values = [h[index][metric] for h in histories]
            entry[metric] = statistics.mean(values)
            entry[metric + '_seeds'] = values
        record['history'].append(entry)
        older = [h['monitor_loss'] for h in record['history'] if h['step'] <= entry['step'] - 20000]
        entry['best_improvement_over_patience'] = (min(older) - min(h['monitor_loss'] for h in record['history'])
                                                   if older else None)
    record['status'] = 'complete'
    atomic_json(out / 'plateau.json', record)
    print('All nine runs finished at exactly 105,000 steps', flush=True)


if __name__ == '__main__':
    main()
