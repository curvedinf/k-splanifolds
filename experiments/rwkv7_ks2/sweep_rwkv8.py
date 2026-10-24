"""RWKV7+ROSA budget match, with reduced-rank and full-rank controls on four GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from sweep_plateau import environment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parent
    jobs=[('baseline',567552),('reduced_control',515328),('hybrid_matched',567296),('hybrid_full',619520)]
    (args.out/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
    def worker(item):
        gpu,(mode,count)=item
        group=args.out/mode;out=group/'fixed_both_11'
        out.mkdir(parents=True)
        cmd=[sys.executable,str(root/'run_rwkv8_gpu.py'),'--mode',mode,
             '--data',str(root/'gpu_runs/data'),'--out',str(out),'--variant','inner_ks2_ffn',
             '--width','96','--layers','4','--ffn-charts','64','--seed','11','--steps','15000',
             '--control-update-scale']
        (out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
        print(json.dumps(dict(event='start',gpu=gpu,mode=mode,parameters=count)),flush=True)
        with (out/'run.log').open('w') as log:
            subprocess.run(cmd,env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((out/'results.json').read_text())
        assert result['arguments']['steps']==result['history'][-1]['step']==15000
        assert result['parameters']==count and result['invalid_steps']==0 and result['ffn_types']==['ks2']*4
        for name,expected in result['source_hashes'].items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
        print(json.dumps(dict(event='trained',gpu=gpu,mode=mode,evaluation=result['evaluation'])),flush=True)
        tiled=group/'tiled_11.json'
        with (out/'tiled.log').open('w') as log:
            subprocess.run([sys.executable,str(root/'evaluate_rwkv8.py'),'--mode',mode,'--runs',str(group),
                '--arm','fixed_both','--seeds','11','--output',str(tiled)],env=environment(gpu),
                stdout=log,stderr=subprocess.STDOUT,check=True)
        result['tiled_evaluation']=json.loads(tiled.read_text())[0]
        result['physical_gpu']=gpu
        (out/'completed.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(event='complete',gpu=gpu,mode=mode,loss=result['tiled_evaluation']['loss'])),flush=True)
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(worker,enumerate(jobs)))
    (args.out/'all_results.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
