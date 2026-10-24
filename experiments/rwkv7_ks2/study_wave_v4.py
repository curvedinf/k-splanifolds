"""Four GPU study queue with source snapshots and horizon/context-aware evaluation."""
import argparse,json,hashlib,subprocess,sys,time
from pathlib import Path
from queue import Queue,Empty
from concurrent.futures import ThreadPoolExecutor
from sweep_plateau import environment

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);a=p.parse_args()
 spec=json.loads(a.manifest.read_text());root=Path(__file__).resolve().parent;out=Path(spec['out']);out.mkdir(parents=True,exist_ok=False)
 queue=Queue()
 for job in spec['jobs']:queue.put(job)
 def worker(gpu):
  rows=[]
  while time.time()<spec['deadline_epoch']:
   try:job=queue.get_nowait()
   except Empty:break
   cfg=job['config'];seed=job.get('seed',11);steps=job.get('steps',15000)
   dest=out/job['id'];dest.mkdir();config=dest/'config.json';config.write_text(json.dumps(cfg,indent=2)+'\n')
   cmd=[sys.executable,str(root/'study_train_v4.py'),'--config',str(config),'--data',str(root/'gpu_runs/data'),'--out',str(dest),'--variant','inner_ks2_ffn',
        '--width',str(cfg.get('width',96)),'--layers','4','--ffn-charts',str(cfg.get('charts',64)),'--seed',str(seed),'--steps',str(steps),
        '--batch-size',str(job.get('batch_size',4)),'--sequence-length',str(job.get('sequence_length',64))]
   (dest/'command.json').write_text(json.dumps(cmd,indent=2)+'\n');print(json.dumps(dict(event='start',time=time.time(),gpu=gpu,job=job)),flush=True)
   try:
    with (dest/'run.log').open('w') as log:subprocess.run(cmd,env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((dest/'results.json').read_text());assert result['history'][-1]['step']==steps and result['invalid_steps']==0
    assert result['parameters']==result['parameter_breakdown']['total']
    for name,h in result['source_hashes'].items():assert hashlib.sha256((dest/'sources'/name.replace('../','parent_')).read_bytes()).hexdigest()==h
    with (dest/'tiled.log').open('w') as log:subprocess.run([sys.executable,str(root/'study_full_eval_v4.py'),'--run',str(dest)],env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,check=True)
    result['intermediate_evaluations']=[]
    for cp in sorted(dest.glob('checkpoint_*.pt')):
     evaluation_path=dest/(cp.stem+'_tiled.json')
     with (dest/(cp.stem+'_tiled.log')).open('w') as log:subprocess.run([sys.executable,str(root/'study_full_eval_v4.py'),'--run',str(dest),'--checkpoint',str(cp),'--output',str(evaluation_path)],env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,check=True)
     result['intermediate_evaluations'].append(json.loads(evaluation_path.read_text()))
    result['context_evaluations']=[]
    for length in job.get('eval_lengths',[]):
     evaluation_path=dest/f'context{length}_tiled.json'
     with (dest/f'context{length}_tiled.log').open('w') as log:subprocess.run([sys.executable,str(root/'study_full_eval_v4.py'),'--run',str(dest),'--sequence-length',str(length),'--output',str(evaluation_path)],env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,check=True)
     result['context_evaluations'].append(json.loads(evaluation_path.read_text()))
    result.update(tiled_evaluation=json.loads((dest/'tiled.json').read_text()),physical_gpu=gpu,study_job_id=job['id'],status='complete')
    (dest/'completed.json').write_text(json.dumps(result,indent=2)+'\n');rows.append(result)
    print(json.dumps(dict(event='complete',time=time.time(),gpu=gpu,job=job['id'],loss=result['tiled_evaluation']['loss'],parameters=result['parameters'],ms=result['training_seconds']*1000/steps)),flush=True)
   except Exception as error:
    failure=dict(job=job,error=repr(error),status='failed',gpu=gpu);(dest/'failure.json').write_text(json.dumps(failure,indent=2)+'\n');rows.append(failure);print(json.dumps(failure),flush=True)
  return rows
 with ThreadPoolExecutor(max_workers=4) as pool:rows=[r for batch in pool.map(worker,range(4)) for r in batch]
 (out/'all_results.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
