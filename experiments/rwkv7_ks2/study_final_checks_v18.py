"""Predeclared final holdout and interleaved speed tests after study training.

Holdout results are only produced after the extended exploration deadline.
GPU 0 handles speed tests; GPUs 1-3 score all predeclared long-run candidates.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from queue import Empty, Queue
import random
import statistics
import subprocess
import sys
import time
from sweep_plateau import environment


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    code = Path(__file__).resolve().parent
    deadline = json.loads((a.root/'study_state.json').read_text())['deadline_epoch']
    while time.time() < deadline or not (a.root/'training_complete.json').exists():
        time.sleep(5)
    out = a.root/'final_checks'
    out.mkdir(exist_ok=False)
    spec = json.loads((a.root/'wave8_long_manifest.json').read_text())
    queue = Queue()
    missing = []
    for job in spec['jobs']:
        path = a.root/'wave8_long'/job['id']
        if (path/'completed.json').exists():
            queue.put((path,'long',job['id']))
        else:
            missing.append(job['id'])
    short_manifest=a.root/'wave9_geometry_learning_manifest.json'
    if short_manifest.exists():
        for job in json.loads(short_manifest.read_text())['jobs']:
            path=a.root/'wave9_geometry_learning'/job['id']
            if (path/'completed.json').exists():queue.put((path,'short',job['id']))
        for seed in (11,22,33):
            reference=a.root/f'initial/mlp/fixed_both_{seed}' if seed!=33 else a.root/'wave3/mlp_seed33'
            queue.put((reference,'short',f'original_mlp_{seed}'))
            frozen=a.root/'wave4_activation/spine_frozen' if seed==11 else a.root/f'wave6_budget/spine_frozen_{seed}'
            queue.put((frozen,'short',f'frozen_spine_{seed}'))
            queue.put((a.root/f'wave4_activation/spine_fused_{seed}','short',f'learned_spine_{seed}'))
    extra_plan=a.root/'extended_holdout_plan.json'
    if extra_plan.exists():
        for name in json.loads(extra_plan.read_text())['manifests']:
            extra=json.loads((a.root/name).read_text())
            for job in extra['jobs']:
                path=Path(extra['out'])/job['id']
                phase='extended_long' if job.get('steps',15000)>=100000 else 'extended_short'
                if (path/'completed.json').exists():queue.put((path,phase,path.parent.name+'/'+job['id']))
                else:missing.append(path.parent.name+'/'+job['id'])
    final_plan=json.loads((a.root/'final_holdout_plan.json').read_text())
    holdout = a.root/final_plan['bytes']
    holdout_sha = hashlib.sha256(holdout.read_bytes()).hexdigest()

    def evaluate(gpu):
        rows = []
        while True:
            try:
                path,phase,label = queue.get_nowait()
            except Empty:
                return rows
            dest = path/final_plan['evaluation_filename']
            with dest.with_suffix('.log').open('w') as log:
                subprocess.run([sys.executable, str(code/'study_full_eval_v18.py'),
                                '--run', str(path), '--validation', str(holdout),
                                '--output', str(dest)], env=environment(gpu),
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            row = dict(job=label,phase=phase,run=str(path.relative_to(a.root)),evaluation=json.loads(dest.read_text()))
            rows.append(row)
            print(json.dumps(dict(event='reserved_holdout', **row)), flush=True)

    def benchmark():
        with (out/'history_prune_check.log').open('w') as log:
            subprocess.run([sys.executable,str(code/'check_history_final_prune.py'),
                            '--run',str(a.root/'wave8_long/original_history_11'),
                            '--out',str(out/'history_prune_check.json'),'--device','cuda'],
                           env=environment(0),stdout=log,stderr=subprocess.STDOUT,check=True)
        configs = [dict(name='original_mlp', family='mlp'),
                   dict(name='frozen_spine46', family='dense_ks2', charts=46, cell='ks2_relu2',
                        geometry_terms='spine', fused_partial=True, frozen_geometry=True),
                   dict(name='history_pruned_fused',family='history',fused_geometry=True,prune_final_side=True)]
        for fused in (False, True):
            suffix = 'fused' if fused else 'eager'
            configs.extend([
                dict(name='history_'+suffix, family='history', fused_geometry=fused),
                dict(name='full31_'+suffix, family='dense_ks2', charts=31,
                     cell='ks2_relu2', fused_geometry=fused),
                dict(name='spine46_'+suffix, family='dense_ks2', charts=46,
                     cell='ks2_relu2', geometry_terms='spine', fused_partial=fused),
                dict(name='undamped33_'+suffix, family='dense_ks2', charts=33,
                     cell='ks2_relu2', geometry_terms='undamped', fused_partial=fused),
            ])
        additional_speed=a.root/'additional_speed_configs.json'
        if additional_speed.exists():configs.extend(json.loads(additional_speed.read_text())['configs'])
        rows = []
        rng = random.Random(7901)
        for repeat in range(5):
            order = configs.copy()
            rng.shuffle(order)
            for cfg in order:
                dest = out/f"speed_{repeat}_{cfg['name']}"
                dest.mkdir()
                config = dest/'config.json'
                config.write_text(json.dumps(cfg, indent=2)+'\n')
                cmd = [sys.executable, str(code/'study_train_v18.py'), '--config', str(config),
                       '--data', str(code/'gpu_runs/data'), '--out', str(dest),
                       '--variant', 'inner_ks2_ffn', '--width', '96', '--layers', '4',
                       '--seed', '11', '--steps', '0']
                with (dest/'run.log').open('w') as log:
                    subprocess.run(cmd, env=environment(0), stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
                result = json.loads((dest/'benchmark.json').read_text())
                row = dict(model=cfg['name'], repeat=repeat, gpu=0,
                           ms=result['benchmark_ms_per_step'], config=cfg)
                rows.append(row)
                print(json.dumps(dict(event='interleaved_speed', **row)), flush=True)
                (out/'speed_raw.json').write_text(json.dumps(rows, indent=2)+'\n')
        summary = []
        for cfg in configs:
            values = [r['ms'] for r in rows if r['model']==cfg['name']]
            summary.append(dict(model=cfg['name'], median_ms=statistics.median(values),
                                min_ms=min(values), max_ms=max(values), values=values))
        result = dict(protocol='GPU 0; five randomized-order rounds, fresh process/model per case; '
                      '100 captured training updates after warmup/reset, including batch copies and LR writes; '
                      'batch 4, sequence 64, FP32. Compilation/evaluation excluded.', rows=rows, summary=summary)
        (out/'speed_results.json').write_text(json.dumps(result, indent=2)+'\n')
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        speed_future = pool.submit(benchmark)
        futures = [pool.submit(evaluate, gpu) for gpu in (1, 2, 3)]
        rows = [r for future in futures for r in future.result()]
        speed_future.result()
    result = dict(holdout_sha256=holdout_sha, holdout_manifest=final_plan['metadata'], missing_training_jobs=missing,
                  selection='All predeclared long-run families; no selection using holdout scores.',
                  rows=rows, finished_epoch=time.time())
    (out/'holdout_results.json').write_text(json.dumps(result, indent=2)+'\n')
    (out/'complete.json').write_text(json.dumps(dict(finished_epoch=time.time(), holdout_runs=len(rows)))+'\n')


if __name__ == '__main__':
    main()
