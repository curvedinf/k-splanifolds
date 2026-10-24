"""Wait for prior four-GPU wave, validate next, run optional kernel check, then train."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sweep_plateau import environment

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--after',type=Path,required=True);a=p.parse_args()
 while not a.after.exists():time.sleep(5)
 root=Path(__file__).resolve().parent;logs=a.manifest.parent/(a.manifest.stem+'_checks_'+str(int(time.time())));logs.mkdir()
 def check(gpu):
  if gpu==0:
   with (logs/'geometry.log').open('w') as f:
    code=subprocess.run([sys.executable,str(root/'check_geometry_gpu.py'),'--out',str(logs/'geometry.json')],env=environment(gpu),stdout=f,stderr=subprocess.STDOUT).returncode
   print(json.dumps(dict(event='geometry_check',exit_code=code,path=str(logs))),flush=True)
  with (logs/f'gpu{gpu}.log').open('w') as f:
   subprocess.run([sys.executable,str(root/'check_study_v2.py'),'--manifest',str(a.manifest),'--shard',str(gpu)],env=environment(gpu),stdout=f,stderr=subprocess.STDOUT,check=True)
  print(json.dumps(dict(event='validation_complete',gpu=gpu)),flush=True)
 with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(check,range(4)))
 subprocess.run([sys.executable,str(root/'study_wave_v2.py'),'--manifest',str(a.manifest)],check=True)
if __name__=='__main__':main()
