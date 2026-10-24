"""Baseline, all RWKV linears, and twelve isolated replacement types on four GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from queue import Queue,Empty
import subprocess
import sys
from sweep_plateau import environment
from rwkv_mux import MODES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parent
    (args.out/'jobs.json').write_text(json.dumps(dict(modes=MODES,seed=11,steps=15000),indent=2)+'\n')
    queue=Queue()
    for mode in MODES:queue.put(mode)
    def worker(gpu):
        rows=[]
        while True:
            try:mode=queue.get_nowait()
            except Empty:return rows
            group=args.out/mode
            out=group/'fixed_both_11'
            out.mkdir(parents=True)
            cmd=[sys.executable,str(root/'run_rwkv_mux_gpu.py'),'--mode',mode,
                '--data',str(root/'gpu_runs/data'),'--out',str(out),'--variant','inner_ks2_ffn',
                '--width','96','--layers','4','--ffn-charts','64','--seed','11',
                '--steps','15000','--control-update-scale']
            (out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
            print(json.dumps(dict(event='start',gpu=gpu,mode=mode)),flush=True)
            with (out/'run.log').open('w') as log:
                subprocess.run(cmd,env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,check=True)
            result=json.loads((out/'results.json').read_text())
            assert result['arguments']['steps']==result['history'][-1]['step']==15000
            assert result['invalid_steps']==0 and result['ffn_types']==['ks2']*4
            for name,expected in result['source_hashes'].items():
                assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
            print(json.dumps(dict(event='trained',gpu=gpu,mode=mode,evaluation=result['evaluation'])),flush=True)
            tiled=group/'tiled_11.json'
            with (out/'tiled.log').open('w') as log:
                subprocess.run([sys.executable,str(root/'evaluate_rwkv_mux.py'),'--mode',mode,
                    '--runs',str(group),'--arm','fixed_both','--seeds','11','--output',str(tiled)],
                    env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,check=True)
            result['tiled_evaluation']=json.loads(tiled.read_text())[0]
            result['physical_gpu']=gpu
            rows.append(result)
            (out/'completed.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(dict(event='complete',gpu=gpu,mode=mode,loss=result['tiled_evaluation']['loss'])),flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=[row for rows in pool.map(worker,range(4)) for row in rows]
    assert len(results)==len(MODES)==14
    (args.out/'all_results.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
