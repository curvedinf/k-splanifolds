"""Fresh paired MLP and selected KS2 baselines for the four-hour study."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from sweep_plateau import environment

root=Path(__file__).resolve().parent
out=root/'gpu_runs/goal_20261022_232213'
jobs=[('mlp',11),('mlp',22),('history',11),('history',22)]
def worker(item):
 gpu,(mode,seed)=item
 group=out/'initial'/mode;dest=group/f'fixed_both_{seed}';dest.mkdir(parents=True,exist_ok=False)
 runner='run_spread_gpu.py' if mode=='mlp' else 'run_side_history_gpu.py'
 extras=['--coding','fixed_both'] if mode=='mlp' else ['--side-width','2048','--control-update-scale']
 cmd=[sys.executable,str(root/runner),*extras,'--data',str(root/'gpu_runs/data'),'--out',str(dest),
      '--variant','mlp' if mode=='mlp' else 'inner_ks2_ffn','--width','96','--layers','4','--ffn-charts','64',
      '--seed',str(seed),'--steps','15000']
 (dest/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 print(json.dumps(dict(event='start',gpu=gpu,mode=mode,seed=seed)),flush=True)
 with (dest/'run.log').open('w') as log:subprocess.run(cmd,env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,check=True)
 evaluator='evaluate_spread.py' if mode=='mlp' else 'evaluate_side_history.py'
 extras=[] if mode=='mlp' else ['--side-width','2048']
 tiled=group/f'tiled_{seed}.json'
 with (dest/'tiled.log').open('w') as log:
  subprocess.run([sys.executable,str(root/evaluator),*extras,'--runs',str(group),'--arm','fixed_both',
    '--seeds',str(seed),'--output',str(tiled)],env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,check=True)
 r=json.loads((dest/'results.json').read_text());r['tiled_evaluation']=json.loads(tiled.read_text())[0]
 r.update(study_mode=mode,physical_gpu=gpu)
 (dest/'completed.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(dict(event='complete',gpu=gpu,mode=mode,seed=seed,loss=r['tiled_evaluation']['loss'])),flush=True)
 return r
with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(worker,enumerate(jobs)))
(out/'initial/results.json').write_text(json.dumps(rows,indent=2)+'\n')
