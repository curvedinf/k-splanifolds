"""Post-training GS128 INT8 FFN ablation; calibration uses training tokens only."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch import nn
from experiments.nanogpt_ks2.bench_forward import load_checkpoint, validate, documents
from experiments.nanogpt_ks2.run import batches
from experiments.common.ks2_cpu.inference import KS2VNNI, _state, _dequantize


class NativeInt8(nn.Module):
    def __init__(self, source, threads, projection_weights=False):
        super().__init__()
        self.cell=KS2VNNI(source,threads)
        if projection_weights:
            for name in ('down','up'):
                w=getattr(self.cell,name)
                w.copy_(_dequantize(w))

    def forward(self,x):
        return self.cell(x.reshape(-1,x.shape[-1])).reshape(x.shape)


@torch.no_grad()
def quantize_geometry_weights(model):
    for ff in model.ff.values():
        state=_state(ff)
        longitudinal=_dequantize(state['geo_t'])
        ff.geo_spine.copy_(longitudinal[:,:,:8])
        ff.geo_t_basis_edge.copy_(longitudinal[:,:,8:].reshape_as(ff.geo_t_basis_edge))
        ff.geo_x_basis_edge.copy_(_dequantize(state['geo_x']).permute(0,2,1,3))
    return model


@torch.inference_mode()
def calibrate(model,stream):
    captured={key:[] for key in model.ff}
    handles=[ff.register_forward_pre_hook(lambda module,args,key=key:captured[key].append(args[0].detach().clone()))
             for key,ff in model.ff.items()]
    try:
        for tokens in stream:
            x=tokens[:-3];model(x,80,176,documents(x))
    finally:
        for h in handles:h.remove()
    for key,ff in model.ff.items():ff.calibrate(captured[key],margin=1.2)
    return {key:{name:getattr(ff,'range_'+name).tolist() for name in ('basis','atten','ex')} for key,ff in model.ff.items()}


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoints',type=Path,required=True)
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--fused-checkpoint',type=Path)
    ap.add_argument('--threads',type=int,default=8)
    args=ap.parse_args()
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    train=np.memmap(args.data_dir/'fineweb_train_000001.bin',dtype='<u2',mode='r',offset=1024)
    val=np.memmap(args.data_dir/'fineweb_val_000000.bin',dtype='<u2',mode='r',offset=1024)
    validation=[torch.from_numpy(np.array(val[i:i+129],dtype=np.int64)) for i in range(0,2048,128)]
    paths=[(f'original_seed_{seed}',seed,args.checkpoints/f'ks2_{seed}.pt') for seed in (11,22,33)]
    if args.fused_checkpoint:paths.append(('fused_training_seed_11',11,args.fused_checkpoint))
    rows=[];calibration=[]
    for label,seed,path in paths:
        model,checkpoint=load_checkpoint(path)
        fp32=validate(model,validation)
        recorded=checkpoint['history'][-1]['val_loss']
        if abs(fp32-recorded)>1e-6:raise RuntimeError('FP32 checkpoint reconstruction mismatch')
        del checkpoint
        stream,offset=batches(train,32,128,seed)
        ranges=calibrate(model,stream)
        calibration.append(dict(checkpoint=label,train_offset=offset,ranges=ranges))
        rows.append(dict(checkpoint=label,variant='fp32',val_ce=fp32,delta_ce=0.,recorded_ce_error=fp32-recorded))
        for variant in ('geometry_weights_gs128','native_int8_geometry','native_int8_geometry_projection_weights_gs128'):
            m=copy.deepcopy(model)
            if variant=='geometry_weights_gs128':quantize_geometry_weights(m)
            else:
                for key in list(m.ff):m.ff[key]=NativeInt8(m.ff[key],args.threads,projection_weights=variant.endswith('projection_weights_gs128'))
            start=time.perf_counter();ce=validate(m,validation)
            row=dict(checkpoint=label,variant=variant,val_ce=ce,delta_ce=ce-fp32,evaluation_seconds=time.perf_counter()-start)
            rows.append(row);print(json.dumps(row),flush=True)
            del m
        del model
    result=dict(protocol=dict(precision='FP32 outside quantized FFN terms',group_size=128,quantizer='symmetric signed INT8 -127..127, absmax scale, round-to-nearest',
                 calibration_tokens=4096,calibration_split='training',calibration_margin=1.2,calibration_windows=[80,176],
                 validation_tokens=2048,validation_sequence=128,validation_split='original fixed validation prefix',
                 geometry_weights='per-chart flattened packed longitudinal/edge coefficients; groups of 128',
                 projection_weights='per-output-row groups of 128, padded final group; dequantized FP32 GEMMs, not INT8 GEMMs',
                 native_arithmetic='INT8 VNNI basis/coefficient dots, INT32 edge products and reductions; FP32 coordinates, damping, warps and residual; exported damping/warp constants rounded through FP16',
                 activation_scales='fixed per chart for longitudinal basis, attenuated edges and edge values; Chebyshev edge basis scale 1/127',
                 no_retraining=True,threads=args.threads,torch=torch.__version__),calibration=calibration,results=rows)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
