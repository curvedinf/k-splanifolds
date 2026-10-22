"""TQ1_0 on every learned FFN parameter: matched MLP and KS2 checkpoint losses."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from experiments.nanogpt_ks2.bench_forward import load_checkpoint,validate
from experiments.nanogpt_ks2.quantize_tq1 import quantize
from experiments.nanogpt_ks2.quantize_controls import NAMES
from experiments.nanogpt_ks2.tq1 import roundtrip


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoints',type=Path,required=True)
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--fused-checkpoint',type=Path)
    ap.add_argument('--ks2-only',action='store_true')
    ap.add_argument('--gs128',action='store_true')
    args=ap.parse_args()
    quantizer=roundtrip
    if args.gs128:
        from experiments.nanogpt_ks2.tq1_gs128 import roundtrip as quantizer
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    data=np.memmap(args.data_dir/'fineweb_val_000000.bin',dtype='<u2',mode='r',offset=1024)
    validation=[torch.from_numpy(np.array(data[i:i+129],dtype=np.int64)) for i in range(0,2048,128)]
    paths=[(kind,seed,'original',args.checkpoints/f'{kind}_{seed}.pt') for seed in (11,22,33) for kind in (('ks2',) if args.ks2_only else ('mlp','ks2'))]
    if args.fused_checkpoint:paths.append(('ks2',11,'fused_training',args.fused_checkpoint))
    rows=[];diagnostics={}
    for kind,seed,source,path in paths:
        model,ck=load_checkpoint(path)
        baseline=ck['history'][-1]['val_loss'];del ck
        stats={};covered=set()
        if kind=='ks2':
            stats=quantize(model,'tq1_geometry_projections',roundtrip=quantizer)
            for key,ff in model.ff.items():
                covered.update(f'{key}.{name}' for name in ('down.weight','up.weight','geo_spine','geo_t_basis_edge','geo_x_basis_edge'))
                for name,attr in NAMES.items():
                    weight=getattr(ff,attr)
                    view=weight.reshape(weight.shape[0],-1) if name in ('distance','importance') else weight.reshape(1,-1)
                    q,stats[key][attr]=quantizer(view)
                    weight.copy_(q.reshape_as(weight));covered.add(f'{key}.{attr}')
        else:
            for key,ff in model.ff.items():
                stats[key]={}
                for name,p in ff.named_parameters():
                    if p.ndim!=2:raise ValueError('Unexpected MLP parameter layout')
                    q,stats[key][name]=quantizer(p);p.copy_(q);covered.add(f'{key}.{name}')
        expected=set(dict(model.ff.named_parameters()))
        if covered!=expected:raise AssertionError((expected-covered,covered-expected))
        count=sum(p.numel() for p in model.ff.parameters())
        quantized_count=sum(s['weights'] for ff in stats.values() for s in ff.values())
        assert quantized_count==count
        ce=validate(model,validation)
        if not math.isfinite(ce):raise ValueError('Nonfinite held-out loss')
        row=dict(kind=kind,seed=seed,source=source,fp32_ce=baseline,tq1_ce=ce,delta_ce=ce-baseline,
                 perplexity_increase_percent=100*math.expm1(ce-baseline),ffn_parameters=count,quantized_parameters=quantized_count,
                 packed_ffn_bytes=sum(s['packed_bytes'] for ff in stats.values() for s in ff.values()),
                 zero_fraction=sum((p==0).sum().item() for p in model.ff.parameters())/count)
        rows.append(row);diagnostics[f'{source}_{kind}_{seed}']=stats
        print(json.dumps(row),flush=True);del model
    args.out.write_text(json.dumps(dict(protocol=dict(quantizer=('custom TQ1 GS128 packed roundtrip, 128 weights/28 bytes' if args.gs128 else 'actual llama.cpp TQ1_0 packed roundtrip, 256 weights/54 bytes'),
               source=json.loads(Path(__file__).with_name('tq1_source.json').read_text()),
               scope='every learned FFN parameter; KS2 includes raw damping and warp parameters before their FP32 transforms',
               grouping='KS2 coefficients packed per chart as prior experiments; damping per family/chart; warp per family across charts; both models projections per output row',
               padding=f'each row zero-padded to a multiple of {128 if args.gs128 else 256} and cropped after decode',
               activations_and_arithmetic='FP32; non-FFN parameters unchanged; no native ternary execution',
               validation_tokens=2048,sequence=128,threads=8,baseline='reused recorded FP32 loss; no FP32 reruns',
               calibration=False,retraining=False,verification=('all named FFN parameters covered; exact counts; packed base-three decoder matches direct rounded values' if args.gs128 else 'all named FFN parameters covered; exact counts; independent NumPy decoder matches C decoder')),
               results=rows,diagnostics=diagnostics),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
