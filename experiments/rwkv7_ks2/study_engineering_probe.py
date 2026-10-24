"""Isolated GPU0 compact-code validation/timing after the queued training wave."""
import argparse,json,random,statistics,subprocess,sys,time,traceback
from pathlib import Path
from sweep_plateau import environment


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--after',type=Path,required=True);a=p.parse_args()
 while not a.after.exists():time.sleep(5)
 out=a.root/'compact_code_checks';out.mkdir(exist_ok=False);code=Path(__file__).resolve().parent
 def command(args,log):
  with (out/log).open('w') as f:subprocess.run([sys.executable,*map(str,args)],env=environment(0),stdout=f,stderr=subprocess.STDOUT,check=True)
 try:
  command([code/'check_structured_codes_gpu.py','--out',out/'kernel.json'],'kernel.log')
  command([code/'check_structured_model.py','--run',a.root/'wave8_long/original_history_11','--out',out/'trained_model.json','--device','cuda'],'trained_model.log')
  command([code/'check_structured_model.py','--run',a.root/'wave8_long/original_history_11','--out',out/'trained_model_real.json','--device','cuda','--real-tokens'],'trained_model_real.log')
  configs=[dict(name='history_dense',family='history',fused_geometry=True),dict(name='history_compact',family='history',fused_geometry=True,structured_codes=True),dict(name='history_pruned',family='history',fused_geometry=True,prune_final_side=True)]
  rows=[];rng=random.Random(41023)
  for repeat in range(5):
   order=configs.copy();rng.shuffle(order)
   for cfg in order:
    dest=out/f'{repeat}_{cfg["name"]}';dest.mkdir();(dest/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    command([code/'study_train_v11.py','--config',dest/'config.json','--data',code/'gpu_runs/data','--out',dest,'--variant','inner_ks2_ffn','--width','96','--layers','4','--seed','11','--steps','0'],dest.name+'.log')
    r=json.loads((dest/'benchmark.json').read_text());row=dict(name=cfg['name'],repeat=repeat,ms=r['benchmark_ms_per_step']);rows.append(row);print(json.dumps(row),flush=True)
    (out/'speed_raw.json').write_text(json.dumps(rows,indent=2)+'\n')
  summary=[dict(name=cfg['name'],median_ms=statistics.median(r['ms'] for r in rows if r['name']==cfg['name']),values=[r['ms'] for r in rows if r['name']==cfg['name']]) for cfg in configs]
  (out/'speed_results.json').write_text(json.dumps(dict(protocol='PhysicalGPU0; five randomized rounds, fresh processes, 100captured training updates per model; batch4 length64 FP32.',rows=rows,summary=summary),indent=2)+'\n')
  command([code/'check_general_geometry.py','--root',a.root,'--out',out/'general_geometry_check.json'],'general_geometry_check.log')
  command([code/'study_component_speed.py','--root',a.root,'--out',out/'component_speed.json'],'component_speed.log')
  result=dict(status='PASS',finished_epoch=time.time(),summary=summary)
 except Exception as e:
  result=dict(status='failed',error=repr(e),traceback=traceback.format_exc(),finished_epoch=time.time());print(json.dumps(result),flush=True)
 (a.root/'compact_code_checks_finished.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
