"""Train and evaluate a direct attention-state skip for 15k matched steps."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from sweep_plateau import environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent
    args.out.mkdir(parents=True, exist_ok=False)
    out = args.out / 'fixed_both_11'
    out.mkdir()
    env = environment(0)
    command = [sys.executable, str(root / 'run_raw_mux_skip_gpu.py'),
               '--data', str(root / 'gpu_runs/data'), '--out', str(out),
               '--variant', 'inner_ks2_ffn', '--width', '96', '--layers', '4',
               '--ffn-charts', '64', '--seed', '11', '--steps', '15000', '--control-update-scale']
    (args.out / 'command.json').write_text(json.dumps(command,indent=2)+'\n')
    with (out / 'run.log').open('w') as log:
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    result = json.loads((out / 'results.json').read_text())
    assert result['parameters'] == 543136 and result['invalid_steps'] == 0
    assert result['ffn_types'] == ['ks2']*4 and result['arguments']['attention_skip']
    assert result['arguments']['steps'] == result['history'][-1]['step'] == 15000
    for name,expected in result['source_hashes'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected,name
    with (out / 'tiled.log').open('w') as log:
        subprocess.run([sys.executable,str(root / 'evaluate_raw_mux_skip.py'),
                        '--runs',str(args.out),'--arm','fixed_both','--seeds','11'],
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    tiled = json.loads((args.out / 'tiled_fixed_both.json').read_text())[0]
    comparisons = {}
    for name,file in [('raw_mux_no_skip','raw_mux'),('demuxed_history','side_history'),
                      ('side_stream_no_history','side_stream_2048_c64')]:
        baseline = json.loads((root / (file+'_summary.json')).read_text())
        previous = json.loads((root / (file+'_results.json')).read_text())
        assert result['data'] == previous['data']
        comparisons[name] = dict(parameters=baseline['parameters'],
            full_shard_loss=baseline['tiled_evaluation']['loss'],
            loss_delta=tiled['loss']-baseline['tiled_evaluation']['loss'],
            benchmark_ms_per_step=baseline['benchmark_ms_per_step'],
            sustained_ms_per_step=baseline['sustained_ms_per_step'])
    summary = dict(architecture=result['architecture'],charts=64,rank=8,layers=4,
        attention_width=96,attention_input_widths=[96,704,1312,1920],
        ffn_input_widths=[192,800,1408,2016],raw_output_width=512,final_demux_width=96,
        attention_skip_coefficient=1,steps=15000,seed=11,parameters=result['parameters'],
        parameter_breakdown=result['parameter_breakdown'],
        benchmark_ms_per_step=result['benchmark_ms_per_step'],
        sustained_ms_per_step=result['training_seconds']/15,
        evaluation=result['evaluation'],tiled_evaluation=tiled,comparisons=comparisons,
        protocol='Same seed/data/15k-step FP32 schedule and physical GPU0 as raw-mux baseline. '
                 'Only architectural change: layers 2-4 add previous post-attention 96D state to their '
                 'Hadamard view before LayerNorm/RWKV; both terms also bypass RWKV in its residual addition. '
                 'Previous attention remains in history as well. No detach or trainable skip coefficient. '
                 'All initial parameters and buffers are identical to the raw-mux baseline; initial '
                 'functions differ because of the new skip. Same raw 512D KS2 history and final-only demux. '
                 'One seed; full-shard validation uses four overlapping offsets on the same held-out shard.')
    (args.out / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root / 'raw_mux_skip_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root / 'raw_mux_skip_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
