"""Run a manifest across four GPUs; never start a new test after the study deadline."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from queue import Queue,Empty
import subprocess
import sys
import time
from sweep_plateau import environment


def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);args=p.parse_args()
 spec=json.loads(args.manifest.read_text());root=Path(__file__).resolve().parent
 out=Path(spec['out']);out.mkdir(parents=True,exist_ok=False)
 jobs=spec['jobs'];queue=Queue()
 for job in jobs:queue.put(job)
 def worker(gpu):
  rows=[]
  while True:
   if time.time()>=spec['deadline_epoch']:return rows
   try:job=queue.get_nowait()
   except Empty:return rows
   cfg=job['config'];seed=job.get('seed',11);steps=job.get('steps',15000)
   group=out/job['id'];dest=group/f'fixed_both_{seed}';dest.mkdir(parents=True)
   config=group/'config.json';config.write_text(json.dumps(cfg,indent=2)+'\n')
   cmd=[sys.executable,str(root/'study_run_gpu.py'),'--config',str(config),'--data',str(root/'gpu_runs/data'),
        '--out',str(dest),'--variant','inner_ks2_ffn','--width','96','--layers','4','--ffn-charts',str(cfg.get('charts',64)),
        '--seed',str(seed),'--steps',str(steps),'--control-update-scale']
   (dest/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
   print(json.dumps(dict(event='start',time=time.time(),gpu=gpu,job=job['id'],config=cfg)),flush=True)
   try:
    with (dest/'run.log').open('w') as log:subprocess.run(cmd,env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((dest/'results.json').read_text())
    assert result['arguments']['steps']==result['history'][-1]['step']==steps and result['invalid_steps']==0
    assert result['parameters']==result['parameter_breakdown']['total']
    for name,expected in result['source_hashes'].items():
     file=dest/'sources'/name
     if not file.exists():file=root/name
     assert hashlib.sha256(file.read_bytes()).hexdigest()==expected,name
    tiled=group/f'tiled_{seed}.json'
    with (dest/'tiled.log').open('w') as log:
     subprocess.run([sys.executable,str(root/'study_evaluate.py'),'--config',str(config),'--runs',str(group),
      '--arm','fixed_both','--seeds',str(seed),'--output',str(tiled)],env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,check=True)
    result.update(tiled_evaluation=json.loads(tiled.read_text())[0],physical_gpu=gpu,study_job_id=job['id'],status='complete')
    (dest/'completed.json').write_text(json.dumps(result,indent=2)+'\n')
    rows.append(result)
    print(json.dumps(dict(event='complete',time=time.time(),gpu=gpu,job=job['id'],loss=result['tiled_evaluation']['loss'],parameters=result['parameters'],ms=result['training_seconds']/15)),flush=True)
   except Exception as error:
    failure=dict(job=job,error=repr(error),status='failed',gpu=gpu)
    (dest/'failure.json').write_text(json.dumps(failure,indent=2)+'\n');rows.append(failure)
    print(json.dumps(dict(event='failed',**failure)),flush=True)
 with ThreadPoolExecutor(max_workers=4) as pool:rows=[r for batch in pool.map(worker,range(4)) for r in batch]
 (out/'all_results.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
