"""Validated four-GPU dispatch for arbitrary-horizon/context study runs."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sweep_plateau import environment

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--after',type=Path);p.add_argument('--partial-check',action='store_true');a=p.parse_args()
 if a.after:
  while not a.after.exists():time.sleep(5)
 spec=json.loads(a.manifest.read_text())
 if time.time()>=spec['deadline_epoch']:raise RuntimeError('Study deadline reached; no new tests')
 root=Path(__file__).resolve().parent;logs=a.manifest.parent/(a.manifest.stem+'_checks_'+str(int(time.time())));logs.mkdir()
 (logs/'manifest.json').write_text(a.manifest.read_text())
 def check(gpu):
  if a.partial_check and gpu in (0,1):
   kind,job=('undamped','undamped33_relu') if gpu==0 else ('spine','spine46_relu')
   with (logs/(kind+'_kernel.log')).open('w') as f:
    code=subprocess.run([sys.executable,str(root/'check_partial_geometry.py'),'--kind',kind,'--trained',str(a.manifest.parent/'wave3'/job),'--out',str(logs/(kind+'_kernel.json'))],env=environment(gpu),stdout=f,stderr=subprocess.STDOUT).returncode
   print(json.dumps(dict(event='partial_kernel_check',kind=kind,exit_code=code,path=str(logs))),flush=True)
  with (logs/f'gpu{gpu}.log').open('w') as f:subprocess.run([sys.executable,str(root/'check_study_v9.py'),'--manifest',str(a.manifest),'--shard',str(gpu)],env=environment(gpu),stdout=f,stderr=subprocess.STDOUT,check=True)
  print(json.dumps(dict(event='validation_complete',gpu=gpu)),flush=True)
 with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(check,range(4)))
 assert a.manifest.read_text()==(logs/'manifest.json').read_text(),'Manifest changed during validation'
 subprocess.run([sys.executable,str(root/'study_wave_v9.py'),'--manifest',str(a.manifest)],check=True)
if __name__=='__main__':main()
