"""IQ2_XXS weight-only loss ablation using the actual llama.cpp reference block format."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from experiments.nanogpt_ks2.bench_forward import load_checkpoint,validate
from experiments.nanogpt_ks2.iq2 import roundtrip
from experiments.common.ks2_cpu.inference import _state


@torch.no_grad()
def quantize(model,variant):
    stats={}
    for key,ff in model.ff.items():
        stats[key]={}
        if variant in ('iq2_geometry','iq2_geometry_projections'):
            state=_state(ff)
            t,stats[key]['geo_t']=roundtrip(state['geo_t'])
            x,stats[key]['geo_x']=roundtrip(state['geo_x'])
            ff.geo_spine.copy_(t[:,:,:8])
            ff.geo_t_basis_edge.copy_(t[:,:,8:].reshape_as(ff.geo_t_basis_edge))
            ff.geo_x_basis_edge.copy_(x.permute(0,2,1,3))
        if variant in ('iq2_projections','iq2_geometry_projections'):
            for name in ('down','up'):
                w=getattr(ff,name).weight
                q,stats[key][name]=roundtrip(w);w.copy_(q)
    return stats


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoints',type=Path,required=True)
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--fused-checkpoint',type=Path)
    args=ap.parse_args()
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    val=np.memmap(args.data_dir/'fineweb_val_000000.bin',dtype='<u2',mode='r',offset=1024)
    validation=[torch.from_numpy(np.array(val[i:i+129],dtype=np.int64)) for i in range(0,2048,128)]
    paths=[(f'original_seed_{seed}',args.checkpoints/f'ks2_{seed}.pt') for seed in (11,22,33)]
    if args.fused_checkpoint:paths.append(('fused_training_seed_11',args.fused_checkpoint))
    rows=[];diagnostics={}
    for label,path in paths:
        model,ck=load_checkpoint(path)
        baseline=validate(model,validation)
        if abs(baseline-ck['history'][-1]['val_loss'])>1e-6:raise RuntimeError('FP32 reconstruction mismatch')
        del ck
        rows.append(dict(checkpoint=label,variant='fp32',val_ce=baseline,delta_ce=0.))
        diagnostics[label]={}
        for variant in ('iq2_geometry','iq2_projections','iq2_geometry_projections'):
            m=copy.deepcopy(model)
            diagnostics[label][variant]=quantize(m,variant)
            ce=validate(m,validation)
            row=dict(checkpoint=label,variant=variant,val_ce=ce,delta_ce=ce-baseline)
            rows.append(row);print(json.dumps(row),flush=True)
            del m
        del model
    result=dict(protocol=dict(quantizer='llama.cpp quantize_iq2_xxs; uniform importance weights',
               source=json.loads(Path(__file__).with_name('iq2_source.json').read_text()),
               block_weights=256,subblock_weights=32,block_bytes=66,codebook_entries=256,signs='7-bit constrained sign patterns per 8 values',subblock_scales_bits=4,superblock_scales='FP16',
               layout='packed geometry rows per chart; projection rows per output; zero-pad each row to a multiple of 256, then discard padding after dequantization',
               execution='actual packed IQ2_XXS roundtrip then FP32 PyTorch arithmetic; no native low-bit runtime, INT8 activation quantization or retraining',
               retained_fp32='all activations, geometry damping/warp parameters, attention, embeddings and vocabulary head',
               validation_tokens=2048,sequence=128,threads=8,calibration='none; required importance vector set to ones, including padding',torch=torch.__version__,
               verification='independent NumPy decoder matches C dequantization bitwise for every quantized tensor'),results=rows,diagnostics=diagnostics)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
