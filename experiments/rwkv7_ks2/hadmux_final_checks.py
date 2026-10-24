"""Run the predeclared reserved scoring and controlled timings after training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from queue import Queue, Empty
import random
import statistics as st
import subprocess
import sys
import time
from sweep_plateau import environment


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root;code=Path(__file__).resolve().parent
    state=json.loads((root/'study_state.json').read_text())
    plan=json.loads((root/'final_holdout_plan.json').read_text())
    while time.time()<state['deadline_epoch'] or not (root/'training_complete.json').exists():
        time.sleep(5)
    complete=json.loads((root/'training_complete.json').read_text())
    assert complete['status']=='PASS'
    out=root/'final_checks';out.mkdir(exist_ok=False)
    holdout=root/plan['bytes'];metadata=json.loads((root/plan['metadata']).read_text())
    digest=hashlib.sha256(holdout.read_bytes()).hexdigest();assert digest==metadata['sha256']
    queue=Queue();expected=[]
    for name in plan['manifests']:
        spec=json.loads((root/name).read_text())
        for job in spec['jobs']:
            path=Path(spec['out'])/job['id'];assert (path/'completed.json').exists()
            item=dict(label=path.parent.name+'/'+job['id'],run=str(path),prior=False)
            expected.append(item);queue.put(item)
    for item in plan['prior_references']:
        item=dict(item,prior=True);expected.append(item);queue.put(item)
    save(out/'coverage.json',dict(plan_sha256=hashlib.sha256((root/'final_holdout_plan.json').read_bytes()).hexdigest(),runs=expected,started_epoch=time.time()))

    def command(args, gpu, log):
        with log.open('w') as stream:
            subprocess.run([sys.executable,*map(str,args)],env=environment(gpu),stdout=stream,stderr=subprocess.STDOUT,check=True)

    # A card finishing the previous wave after the launch cutoff may never
    # receive a Pareto training job. Validate those configurations on that
    # now-idle card before final scoring/timing; no training queue is restarted.
    late_checks=[]
    for wave in ('wave4_pareto','wave5_warps'):
        pareto_manifest=root/(wave+'_manifest.json')
        for gpu in range(4):
            check_dir=root/wave/f'validation_gpu{gpu}'
            if check_dir.exists():continue
            check_dir.mkdir()
            (check_dir/'manifest.json').write_bytes(pareto_manifest.read_bytes())
            command([code/'check_hadmux_v2.py','--manifest',pareto_manifest],gpu,check_dir/'check.log')
            late_checks.append(dict(wave=wave,gpu=gpu,finished_epoch=time.time(),reason='Card was not released for this wave before the training-launch cutoff. Numerical validation only, after all training completed.'))
    save(out/'late_gpu_validation.json',dict(checks=late_checks))

    def evaluate(gpu):
        rows=[]
        while True:
            try:item=queue.get_nowait()
            except Empty:return rows
            path=Path(item['run']);dest=out/'reserved'/item['label'];dest.mkdir(parents=True)
            result=dest/'evaluation.json'
            command([code/'hadmux_eval_v2.py','--run',path,'--validation',holdout,'--output',result],gpu,dest/'evaluation.log')
            ev=json.loads(result.read_text());assert ev['validation_sha256']==digest
            row=dict(**item,evaluation=ev,evidence=str(result.relative_to(root)))
            rows.append(row)
            print(json.dumps(dict(event='reserved_score',label=item['label'],loss=ev['loss'],gpu=gpu)),flush=True)

    def benchmark():
        configs=[]
        transfer=json.loads((root/'wave1_transfer_manifest.json').read_text())
        for name in sorted({j['config']['name'] for j in transfer['jobs']}):
            configs.append(next(j['config'] for j in transfer['jobs'] if j['config']['name']==name))
        scaling=json.loads((root/'wave2_scaling_manifest.json').read_text())
        for name in sorted({j['config']['name'] for j in scaling['jobs']}):
            configs.append(next(j['config'] for j in scaling['jobs'] if j['config']['name']==name))
        pareto=json.loads((root/'wave4_pareto_manifest.json').read_text())
        for name in sorted({j['config']['name'] for j in pareto['jobs']}):
            configs.append(next(j['config'] for j in pareto['jobs'] if j['config']['name']==name))
        warps=json.loads((root/'wave5_warps_manifest.json').read_text())
        configs.append(warps['jobs'][0]['config'])
        selection=json.loads((root/'long_selection.json').read_text())
        selected=selection['candidates'][1]
        trained=root/'wave3_long'/('long_'+selected['name']+'_11')
        command([code/'check_hadmux_prune.py','--run',trained,'--out',out/'history_prune_check.json','--device','cuda'],0,out/'history_prune_check.log')
        configs.append(dict(selected['config'],name=selected['name']+'_pruned',prune_final_side=True))
        prune_cases=[]
        for name,wave in [('had_spine_r8_frozen','wave1_transfer'),('pareto_frozen_r8_f1024_s2048','wave4_pareto'),('pareto_frozen_r8_f2048_s2048','wave4_pareto'),('pareto_r8_warps_only_f512_s2048','wave5_warps')]:
            parent=next(cfg for cfg in configs if cfg['name']==name)
            for seed in (11,22,33):
                trained=root/wave/f'{name}_{seed}'
                dest=out/f'prune_{name}_{seed}.json'
                command([code/'check_hadmux_prune.py','--run',trained,'--out',dest,'--device','cuda'],0,dest.with_suffix('.log'))
                checked=json.loads(dest.read_text());assert checked['status']=='PASS'
                prune_cases.append(dict(model=name,seed=seed,check=str(dest.relative_to(root)),**checked))
            configs.append(dict(parent,name=name+'_pruned',prune_final_side=True))
        save(out/'pareto_prune_checks.json',dict(status='PASS',cases=prune_cases,note='Three trained checkpoints per low-cost architecture. Pruning removes only unused final side-output rows and their gains; equivalent quality points reuse the parent checkpoints and are not new independent training runs.'))
        prior_seen=set()
        for reference in plan['prior_references']:
            cfg=json.loads((Path(reference['run'])/'completed.json').read_text()).get('study_config')
            if not cfg or cfg.get('family')=='mlp':continue
            key=json.dumps({k:v for k,v in cfg.items() if k!='name'},sort_keys=True)
            if key in prior_seen:continue
            prior_seen.add(key)
            configs.append(dict(cfg,name='prior_'+cfg['name']))
        save(out/'speed_configs.json',configs)
        rows=[];rng=random.Random(7901)
        for repeat in range(5):
            order=configs.copy();rng.shuffle(order)
            for cfg in order:
                dest=out/f"speed_{repeat}_{cfg['name']}";dest.mkdir()
                save(dest/'config.json',cfg)
                command([code/'hadmux_train_v3.py','--config',dest/'config.json','--data',code/'gpu_runs/data','--out',dest,'--variant','inner_ks2_ffn','--width','96','--layers','4','--seed','11','--steps','0'],0,dest/'run.log')
                measured=json.loads((dest/'benchmark.json').read_text())
                rows.append(dict(model=cfg['name'],repeat=repeat,gpu=0,ms=measured['benchmark_ms_per_step'],config=cfg))
                save(out/'speed_raw.json',rows)
            print(json.dumps(dict(event='controlled_timing_round',repeat=repeat,models=len(configs))),flush=True)
        summary=[]
        for cfg in configs:
            values=[r['ms'] for r in rows if r['model']==cfg['name']]
            summary.append(dict(model=cfg['name'],median_ms=st.median(values),min_ms=min(values),max_ms=max(values),values=values))
        save(out/'speed_results.json',dict(protocol='GPU0; five randomized rounds, fresh process/model per case, 100 captured AdamW updates after warmup/reset, FP32 B4 T64. Includes batch copies and LR writes; excludes compilation/evaluation. Other GPUs perform reserved evaluation independently.',rows=rows,summary=summary))
        command([code/'hadmux_projection_speed.py','--out',root/'projection_speed.json'],0,out/'projection_speed.log')

    with ThreadPoolExecutor(max_workers=4) as pool:
        speed=pool.submit(benchmark)
        evaluation=[pool.submit(evaluate,gpu) for gpu in (1,2,3)]
        rows=[r for future in evaluation for r in future.result()]
        save(out/'holdout_results.json',dict(holdout_sha256=digest,rows=rows,finished_epoch=time.time(),selection=plan['selection']))
        speed.result()
    assert len(rows)==len(expected)
    save(out/'complete.json',dict(status='PASS',finished_epoch=time.time(),holdout_runs=len(rows)))


if __name__=='__main__':main()
