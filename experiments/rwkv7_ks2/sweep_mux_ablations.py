"""Twelve independent 15k-step experiments, dynamically queued across four MI100s."""
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
    root=Path(__file__).parent
    jobs=[dict(mode=m,seed=11,code_seed=103415) for m in
          ('head_skip','output_gains','persistent_state','source_norm')]
    for mode in ('baseline','demuxed_history'):
        jobs += [dict(mode=mode,seed=s,code_seed=103415) for s in (22,33)]
        jobs += [dict(mode=mode,seed=11,code_seed=c) for c in (103416,103417)]
    (args.out/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
    queue=Queue()
    for job in jobs:queue.put(job)
    def worker(gpu):
        rows=[]
        while True:
            try:job=queue.get_nowait()
            except Empty:return rows
            mode,seed,code=job['mode'],job['seed'],job['code_seed']
            group=args.out/f'{mode}_c{code}'
            out=group/f'fixed_both_{seed}'
            out.mkdir(parents=True)
            cmd=[sys.executable,str(root/'run_mux_ablation_gpu.py'),'--mode',mode,'--code-seed',str(code),
                 '--data',str(root/'gpu_runs/data'),'--out',str(out),'--variant','inner_ks2_ffn',
                 '--width','96','--layers','4','--ffn-charts','64','--seed',str(seed),'--steps','15000',
                 '--control-update-scale']
            (out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
            print(json.dumps(dict(event='start',gpu=gpu,**job)),flush=True)
            with (out/'run.log').open('w') as log:
                subprocess.run(cmd,env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,check=True)
            result=json.loads((out/'results.json').read_text())
            assert result['arguments']['steps']==result['history'][-1]['step']==15000
            assert result['invalid_steps']==0 and result['ffn_types']==['ks2']*4
            assert result['arguments']['attention_skip'] is False
            for name,expected in result['source_hashes'].items():
                assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
            print(json.dumps(dict(event='trained',gpu=gpu,**job,evaluation=result['evaluation'])),flush=True)
            tiled=group/f'tiled_{seed}.json'
            with (out/'tiled.log').open('w') as log:
                subprocess.run([sys.executable,str(root/'evaluate_mux_ablation.py'),'--mode',mode,
                                '--code-seed',str(code),'--runs',str(group),'--arm','fixed_both',
                                '--seeds',str(seed),'--output',str(tiled)],env=environment(gpu),
                               stdout=log,stderr=subprocess.STDOUT,check=True)
            result['tiled_evaluation']=json.loads(tiled.read_text())[0]
            result['physical_gpu']=gpu
            rows.append(result)
            (out/'completed.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(dict(event='complete',gpu=gpu,**job,loss=result['tiled_evaluation']['loss'])),flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=[row for rows in pool.map(worker,range(4)) for row in rows]
    assert len(results)==12
    (args.out/'all_results.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
