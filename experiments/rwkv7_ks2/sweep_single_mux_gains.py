"""Three isolated gain ablations and two fresh controls, using four MI100s."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from queue import Queue, Empty
import subprocess
import sys
from sweep_plateau import environment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parent
    jobs=[('dynamic',562736),('output',542432),('input',544032),('baseline',540000),('demuxed_history',567552)]
    (args.out/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
    queue=Queue()
    for job in jobs:queue.put(job)
    def worker(gpu):
        rows=[]
        while True:
            try:mode,count=queue.get_nowait()
            except Empty:return rows
            group=args.out/mode
            out=group/'fixed_both_11'
            out.mkdir(parents=True)
            if mode in ('dynamic','input','output'):
                runner,evaluator='run_single_mux_gains_gpu.py','evaluate_single_mux_gains.py'
                extra=['--gain-placement',mode]
            elif mode=='baseline':
                runner,evaluator='run_single_mux_gpu.py','evaluate_single_mux.py'
                extra=[]
            else:
                runner,evaluator='run_side_history_gpu.py','evaluate_side_history.py'
                extra=['--side-width','2048']
            cmd=[sys.executable,str(root/runner),*extra,'--data',str(root/'gpu_runs/data'),
                 '--out',str(out),'--variant','inner_ks2_ffn','--width','96','--layers','4',
                 '--ffn-charts','64','--seed','11','--steps','15000','--control-update-scale']
            (out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
            print(json.dumps(dict(event='start',gpu=gpu,mode=mode,parameters=count)),flush=True)
            with (out/'run.log').open('w') as log:
                subprocess.run(cmd,env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,check=True)
            result=json.loads((out/'results.json').read_text())
            assert result['parameters']==count
            assert result['arguments']['steps']==result['history'][-1]['step']==15000
            assert result['invalid_steps']==0 and result['ffn_types']==['ks2']*4
            for name,expected in result['source_hashes'].items():
                assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
            print(json.dumps(dict(event='trained',gpu=gpu,mode=mode,evaluation=result['evaluation'])),flush=True)
            tiled=group/'tiled_11.json'
            with (out/'tiled.log').open('w') as log:
                subprocess.run([sys.executable,str(root/evaluator),*extra,'--runs',str(group),
                    '--arm','fixed_both','--seeds','11','--output',str(tiled)],env=environment(gpu),
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            result['tiled_evaluation']=json.loads(tiled.read_text())[0]
            result.update(physical_gpu=gpu,gain_ablation=mode)
            rows.append(result)
            (out/'completed.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(dict(event='complete',gpu=gpu,mode=mode,loss=result['tiled_evaluation']['loss'])),flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=[row for rows in pool.map(worker,range(4)) for row in rows]
    assert len(results)==5
    (args.out/'all_results.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
