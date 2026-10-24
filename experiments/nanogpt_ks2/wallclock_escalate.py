"""After a replicated wall-clock win, retime the untouched MLP at four times the steps."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--steps',type=int,required=True)
    ap.add_argument('--candidates',type=Path,nargs=3,required=True)
    ap.add_argument('--record-only',action='store_true',
                    help='Compare completed runs without launching the next control')
    args=ap.parse_args()
    study_path=args.root/'study.json'
    deadline=datetime.fromisoformat(json.loads(study_path.read_text())['stop_new_experiments'])
    paths=[args.root/f'stage{args.steps}_control'/f'mlp{args.steps}_{seed}'/'result.json'
           for seed in (11,22,33)]
    while not all(path.exists() for path in paths+args.candidates):
        if args.record_only:
            raise FileNotFoundError('Record-only mode requires all six completed results')
        if datetime.now(timezone.utc)>=deadline:
            print('Deadline reached before a replicated comparison was available',flush=True)
            return
        time.sleep(5)
    controls=[json.loads(p.read_text()) for p in paths]
    candidates=[json.loads(p.read_text()) for p in args.candidates]
    assert {x['seed'] for x in controls}=={x['seed'] for x in candidates}=={11,22,33}
    assert all(x['kind']=='mlp' and x['steps']==args.steps for x in controls)
    assert all(x['kind']=='ks3' and 741400<=x['counts']['ffn']<=743424 for x in candidates)
    configs=[{k:v for k,v in x['candidate'].items() if k!='wall_budget'} for x in candidates]
    assert all(x==configs[0] for x in configs)
    target=dict(steps=args.steps,loss=statistics.mean(x['final_val_loss'] for x in controls),
                wall_budget=statistics.mean(x['seconds'] for x in controls),
                seeds=[{k:x[k] for k in ('seed','final_val_loss','seconds')} for x in controls])
    save(args.root/f'target{args.steps}.json',target)
    by_seed={x['seed']:x for x in controls}
    win=all(x['seconds']<target['wall_budget'] and
            x['final_val_loss']<by_seed[x['seed']]['final_val_loss'] for x in candidates)
    comparison=dict(target=target,candidate=configs[0],replicated_win=win,
                    mean_loss=statistics.mean(x['final_val_loss'] for x in candidates),
                    mean_seconds=statistics.mean(x['seconds'] for x in candidates),
                    sources=[str(p) for p in args.candidates],
                    seeds=[{k:x[k] for k in ('seed','final_val_loss','seconds','counts')} for x in candidates])
    save(args.root/f'comparison{args.steps}.json',comparison)
    print(json.dumps(comparison),flush=True)
    if not win:return
    save(args.root/f'milestone{args.steps}.json',comparison)
    if args.record_only:
        print('Win recorded; no new control requested in record-only mode',flush=True)
        return
    if datetime.now(timezone.utc)>=deadline:
        print('Win recorded; deadline prevents a new control run',flush=True)
        return
    next_steps=4*args.steps
    out=args.root/f'stage{next_steps}_control'
    if out.exists():raise RuntimeError(f'Next control already exists: {out}')
    jobs=[dict(name=f'mlp{next_steps}',seed=s,
               config=dict(kind='mlp',steps=next_steps,wall_budget=None)) for s in (11,22,33)]
    job_path=args.root/f'stage{next_steps}_jobs.json';save(job_path,jobs)
    study=json.loads(study_path.read_text())
    study['active_target']=dict(steps=next_steps,loss=None,wall_budget=None,status='retiming unchanged MLP')
    save(study_path,study)
    print(f'Starting unchanged MLP at {next_steps} steps on GPUs 0, 1, 2',flush=True)
    subprocess.run([sys.executable,'-m','experiments.nanogpt_ks2.wallclock_wave',
                    '--jobs',str(job_path),'--out',str(out),'--study',str(study_path),
                    '--gpus','0','1','2'],check=True)


if __name__=='__main__':main()
