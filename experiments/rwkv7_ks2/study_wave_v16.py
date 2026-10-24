"""Four GPU study queue with source snapshots and horizon/context-aware evaluation."""
import argparse,json,hashlib,subprocess,sys,time
from pathlib import Path
from queue import Queue,Empty
from concurrent.futures import ThreadPoolExecutor
from sweep_plateau import environment

def available_gpus(previous_manifest,previous_log):
 spec=json.loads(previous_manifest.read_text())
 required={job['id'] for job in spec['jobs']};started=set();active={}
 for line in previous_log.read_text().splitlines():
  try:event=json.loads(line)
  except ValueError:continue
  if event.get('event')=='start':
   job=event['job']['id'];started.add(job);active[event['gpu']]=job
  elif event.get('event')=='complete':
   if active.get(event['gpu'])==event['job']:del active[event['gpu']]
  elif event.get('status')=='failed' and 'gpu' in event:
   if active.get(event['gpu'])==event['job']['id']:del active[event['gpu']]
 # A card can only be reused once the old queue is empty, including its last dispatch.
 if not required.issubset(started):return set()
 return set(range(4))-set(active)

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--after-manifest',type=Path);p.add_argument('--after-log',type=Path);p.add_argument('--gpu0-after',type=Path);p.add_argument('--requires-check',type=Path);a=p.parse_args()
 assert bool(a.after_manifest)==bool(a.after_log)
 spec=json.loads(a.manifest.read_text());root=Path(__file__).resolve().parent
 if a.requires_check:
  while time.time()<spec['deadline_epoch'] and not a.requires_check.exists():time.sleep(5)
  if not a.requires_check.exists():raise RuntimeError('Required kernel check unavailable before deadline')
  checked=json.loads(a.requires_check.read_text());assert checked['status']=='PASS'
  assert checked['source_sha256']==hashlib.sha256((root/'ks2_general_gpu.py').read_bytes()).hexdigest(),'Kernel changed after numerical validation'
 out=Path(spec['out']);out.mkdir(parents=True,exist_ok=False)
 queue=Queue()
 for job in spec['jobs']:queue.put(job)
 def worker(gpu):
  rows=[]
  if gpu==0 and a.gpu0_after:
   while time.time()<spec['deadline_epoch'] and not a.gpu0_after.exists():time.sleep(5)
  if a.after_manifest:
   while time.time()<spec['deadline_epoch'] and gpu not in available_gpus(a.after_manifest,a.after_log):time.sleep(5)
   if time.time()>=spec['deadline_epoch']:return rows
   print(json.dumps(dict(event='gpu_released_by_previous_wave',time=time.time(),gpu=gpu)),flush=True)
   check_dir=out/f'validation_gpu{gpu}';check_dir.mkdir()
   manifest_text=a.manifest.read_text();(check_dir/'manifest.json').write_text(manifest_text)
   with (check_dir/'check.log').open('w') as log:
    subprocess.run([sys.executable,str(root/'check_study_v15.py'),'--manifest',str(a.manifest),'--shard','0','--stride','1'],env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,check=True)
   assert a.manifest.read_text()==manifest_text,'Manifest changed during validation'
   print(json.dumps(dict(event='validation_complete',time=time.time(),gpu=gpu)),flush=True)
  while time.time()<spec['deadline_epoch']:
   try:job=queue.get_nowait()
   except Empty:break
   cfg=job['config'];seed=job.get('seed',11);steps=job.get('steps',15000)
   dest=out/job['id'];dest.mkdir();config=dest/'config.json';config.write_text(json.dumps(cfg,indent=2)+'\n')
   cmd=[sys.executable,str(root/'study_train_v15.py'),'--config',str(config),'--data',str(job.get('data',root/'gpu_runs/data')),'--out',str(dest),'--variant','inner_ks2_ffn',
        '--width',str(cfg.get('width',96)),'--layers','4','--ffn-charts',str(cfg.get('charts',64)),'--seed',str(seed),'--steps',str(steps),
        '--batch-size',str(job.get('batch_size',4)),'--sequence-length',str(job.get('sequence_length',64))]
   (dest/'command.json').write_text(json.dumps(cmd,indent=2)+'\n');print(json.dumps(dict(event='start',time=time.time(),gpu=gpu,job=job)),flush=True)
   try:
    with (dest/'run.log').open('w') as log:subprocess.run(cmd,env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((dest/'results.json').read_text());assert result['history'][-1]['step']==steps and result['invalid_steps']==0
    assert result['parameters']==result['parameter_breakdown']['total']
    for name,h in result['source_hashes'].items():assert hashlib.sha256((dest/'sources'/name.replace('../','parent_')).read_bytes()).hexdigest()==h
    with (dest/'tiled.log').open('w') as log:subprocess.run([sys.executable,str(root/'study_full_eval_v16.py'),'--run',str(dest)],env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,check=True)
    result['intermediate_evaluations']=[]
    for cp in sorted(dest.glob('checkpoint_*.pt')):
     evaluation_path=dest/(cp.stem+'_tiled.json')
     with (dest/(cp.stem+'_tiled.log')).open('w') as log:subprocess.run([sys.executable,str(root/'study_full_eval_v16.py'),'--run',str(dest),'--checkpoint',str(cp),'--output',str(evaluation_path)],env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,check=True)
     result['intermediate_evaluations'].append(json.loads(evaluation_path.read_text()))
    result['context_evaluations']=[]
    for length in job.get('eval_lengths',[]):
     evaluation_path=dest/f'context{length}_tiled.json'
     with (dest/f'context{length}_tiled.log').open('w') as log:subprocess.run([sys.executable,str(root/'study_full_eval_v16.py'),'--run',str(dest),'--sequence-length',str(length),'--output',str(evaluation_path)],env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,check=True)
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
