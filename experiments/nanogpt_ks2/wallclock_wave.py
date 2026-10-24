"""Four-GPU launcher with a recorded deadline and one independent job per GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--jobs',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--study',type=Path,default=Path('experiments/nanogpt_ks2/runs/wallclock_20261023/study.json'))
    ap.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3])
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'launcher.pid').write_text(str(os.getpid())+'\n')
    study=json.loads(args.study.read_text());deadline=datetime.fromisoformat(study['stop_new_experiments'])
    jobs=json.loads(args.jobs.read_text())
    (args.out/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
    def worker(index,gpu):
        records=[]
        for job in jobs[index::len(args.gpus)]:
            if datetime.now(timezone.utc)>=deadline:
                records.append(dict(job=job,status='deadline'));continue
            name=job['name']+'_'+str(job['seed'])
            cfg=args.out/(name+'_config.json');cfg.write_text(json.dumps(job['config'],indent=2)+'\n')
            cmd=[sys.executable,'-m','experiments.nanogpt_ks2.wallclock','--config',str(cfg),
                 '--out',str(args.out/name),'--seed',str(job['seed']),'--gpu',str(gpu)]
            with (args.out/(name+'.log')).open('w') as log:
                p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                (args.out/(name+'.pid')).write_text(str(p.pid)+'\n')
                status=p.wait()
            records.append(dict(job=job,status='complete' if status==0 else 'failed',returncode=status,gpu=gpu))
            print(records[-1],flush=True)
        return records
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures=[pool.submit(worker,i,gpu) for i,gpu in enumerate(args.gpus)]
        records=[r for f in futures for r in f.result()]
    (args.out/'completion.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
