"""Build an auditable flat index of completed study experiments, excluding smoke tests."""
import argparse,csv,json,math,time,shlex
from pathlib import Path

def collect(root):
 rows=[]
 suspended={}
 pause_record=root/'mixed_rank_interleave.json'
 if pause_record.exists():
  for item in json.loads(pause_record.read_text()).get('processes',[]):
   args=shlex.split(item['command'])
   if '--out' in args:
    output=Path(args[args.index('--out')+1]).resolve()
    suspended[str(output)]=item
 for path in sorted(root.rglob('completed.json')):
  r=json.loads(path.read_text());a=r['arguments'];parts=path.relative_to(root).parts;cfg=r.get('study_config')
  if cfg is None:cfg=dict(name=parts[1],family='mlp' if parts[1]=='mlp' else 'history')
  ev=r['tiled_evaluation'];breakdown=r.get('parameter_breakdown',{});geometry=breakdown.get('geometry',breakdown.get('ks2_geometry',0))
  if not breakdown and cfg['family']=='history':
   assert r['ffn_types']==['ks2']*a['layers']
   geometry=a['layers']*a['ffn_charts']*820
  steps=a['steps'];batch=r.get('batch_size',4);length=r.get('sequence_length',64)
  assert r['history'][-1]['step']==steps and r['invalid_steps']==0 and math.isfinite(ev['loss'])
  pause=suspended.get(str(path.parent.resolve()))
  rows.append(dict(wave=parts[0],job=r.get('study_job_id',cfg['name']),family=cfg.get('family','history'),feature=(cfg.get('outer_activation') or cfg.get('cell','ks2')) if geometry or cfg.get('frozen_geometry') else cfg.get('cell','relu2'),
   attention=cfg.get('attention','rwkv7'),geometry_terms=cfg.get('geometry_terms','full' if geometry or cfg.get('frozen_geometry') else 'none'),frozen_geometry=cfg.get('frozen_geometry',False),backend='fused_general' if cfg.get('fused_general') else 'fused_partial' if cfg.get('fused_partial') else 'fused' if cfg.get('fused_geometry') else 'eager',
   data_train_bytes=r['data']['train']['bytes'],data_train_sha256=r['data']['train']['sha256'],data_validation_sha256=r['data']['val']['sha256'],
   seed=a['seed'],steps=steps,batch_size=batch,sequence_length=length,training_targets=r['training_targets'],parameters=r['parameters'],geometry_parameters=geometry,
   full_validation_loss=ev['loss'],validation_target_visits=ev['targets'],expanded_validation_loss=r['evaluation']['expanded_val'],sampled_training_loss=r['evaluation']['sampled_train'],
   training_seconds=r['training_seconds'],ms_per_step=r['training_seconds']*1000/steps,training_targets_per_second=r['training_targets']/r['training_seconds'],
   suspended_for_mixed_screen=bool(pause),scheduler_pause_wall_seconds=pause.get('pause_seconds') if pause else 0.,
   benchmark_ms_per_step=r['benchmark_ms_per_step'],parameter_bytes=r.get('parameter_bytes',4*r['parameters']),buffer_bytes=r.get('buffer_bytes'),peak_allocated_bytes=r.get('peak_allocated_bytes'),
   physical_gpu=r['physical_gpu'],config=cfg,source=str(path.relative_to(root)),checkpoint=str(path.parent.relative_to(root)/'checkpoint.pt'),
   intermediate_evaluations=r.get('intermediate_evaluations',[]),context_evaluations=r.get('context_evaluations',[])))
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();rows=collect(a.root)
 data=dict(generated_epoch=time.time(),metric='Cross entropy in nats per byte; four overlapping validation tilings, not independent test sets.',
           timing_note='ms_per_step and training_targets_per_second retain raw recorded training_seconds. Runs flagged suspended_for_mixed_screen may include scheduler suspension in those times and must not be used as active-speed estimates. No pause duration is silently subtracted. Final controlled timing is in final_checks/speed_results.json.',rows=rows,
           failures=[dict(path=str(p.relative_to(a.root)),**json.loads(p.read_text())) for p in a.root.rglob('failure.json')])
 (a.root/'all_runs.json').write_text(json.dumps(data,indent=2)+'\n')
 if rows:
  keys=[k for k in rows[0] if k not in ('config','intermediate_evaluations','context_evaluations')]+['config']
  with (a.root/'all_runs.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
   for row in rows:w.writerow({k:json.dumps(row[k],sort_keys=True) if k=='config' else row[k] for k in keys})
 print(json.dumps(dict(completed=len(rows),failed=len(data['failures']),by_wave={wave:sum(r['wave']==wave for r in rows) for wave in sorted(set(r['wave'] for r in rows))})))
if __name__=='__main__':main()
