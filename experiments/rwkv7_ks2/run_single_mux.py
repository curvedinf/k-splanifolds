"""Benchmark the prior two-mux model, then train/evaluate the corrected layout."""
import argparse
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
    root=Path(__file__).parent
    env=environment(0)
    benchmarks={}
    for name,worker,steps in [('two_mux','run_raw_mux_gpu.py',0),('single_mux','run_single_mux_gpu.py',15000)]:
        out=args.out/('benchmark_two_mux' if not steps else 'fixed_both_11')
        out.mkdir()
        cmd=[sys.executable,str(root/worker),'--data',str(root/'gpu_runs/data'),'--out',str(out),
             '--variant','inner_ks2_ffn','--width','96','--layers','4','--ffn-charts','64',
             '--seed','11','--steps',str(steps),'--control-update-scale']
        (out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
        with (out/'run.log').open('w') as log:
            subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        record=json.loads((out/('results.json' if steps else 'benchmark.json')).read_text())
        benchmarks[name]=record['benchmark_ms_per_step']
        print(json.dumps(dict(event='complete',name=name,benchmark_ms=benchmarks[name])),flush=True)
    result=record
    assert result['parameters']==540000 and result['invalid_steps']==0
    assert result['ffn_types']==['ks2']*4 and result['arguments']['muxes_per_layer']==1
    assert result['arguments']['steps']==result['history'][-1]['step']==15000
    for name,expected in result['source_hashes'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
    with (out/'tiled.log').open('w') as log:
        subprocess.run([sys.executable,str(root/'evaluate_single_mux.py'),'--runs',str(args.out),
                        '--arm','fixed_both','--seeds','11'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    tiled=json.loads((args.out/'tiled_fixed_both.json').read_text())[0]
    comparisons={}
    for key,file in [('two_mux','raw_mux'),('demuxed_history','side_history'),('side_stream','side_stream_2048_c64')]:
        prior=json.loads((root/(file+'_summary.json')).read_text())
        comparisons[key]=dict(full_shard_loss=prior['tiled_evaluation']['loss'],
            loss_delta=tiled['loss']-prior['tiled_evaluation']['loss'],
            historical_benchmark_ms=prior['benchmark_ms_per_step'],
            historical_sustained_ms=prior['sustained_ms_per_step'],parameters=prior['parameters'])
    summary=dict(architecture=result['architecture'],parameters=result['parameters'],
        parameter_breakdown=result['parameter_breakdown'],steps=15000,seed=11,
        input_widths=[96,704,1312,1920],mux_output_width=608,ks2_coordinates=512,rwkv_view=96,
        final_demux_input=608,final_demux_output=96,
        benchmark_ms_same_gpu=benchmarks,sustained_ms_per_step=result['training_seconds']/15,
        benchmark_speedup_fraction=1-benchmarks['single_mux']/benchmarks['two_mux'],
        evaluation=result['evaluation'],tiled_evaluation=tiled,comparisons=comparisons,
        protocol='One layer-entry mux with parallel 64x8 KS2 coordinates and one 96D RWKV view. '
                 'Shared input normalization; KS2 retains coordinate-wise causal time mix but no input/output projection. '
                 'Attention state bypasses geometry within the full 608D layer residual. Both final branches feed '
                 'the single 608-to-96 demux before logits. Same seed11/data/15k-step FP32 schedule; '
                 'fresh sequential benchmarks on GPU0, warmup/benchmark updates reset. Architecture and '
                 'initial function change from the mistaken two-mux versions. One seed; same held-out shard.')
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/'single_mux_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/'single_mux_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
