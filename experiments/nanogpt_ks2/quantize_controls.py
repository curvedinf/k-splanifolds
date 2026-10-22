"""Isolated raw/transformed damping and warp precision sweeps, with FP32 arithmetic."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
from types import MethodType

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.nn import functional as F
from experiments.nanogpt_ks2.bench_forward import load_checkpoint,validate
from experiments.common.ks2_stabilized import _strength,_predicates
from experiments.common.ks2_cpu.inference import _dequantize
from experiments.nanogpt_ks2 import q4k,iq2,tq1

NAMES={'distance':'geo_den_delta','importance':'geo_den_weight','mean_scale':'geo_raw_scale',
       'mean_shift':'geo_raw_shift','edge_scale':'geo_edge_raw_scale','edge_shift':'geo_edge_raw_shift'}
SCOPES={name:(name,) for name in NAMES}
SCOPES.update(damping=('distance','importance'),warp=('mean_scale','mean_shift','edge_scale','edge_shift'),both=tuple(NAMES))
LEVELS=('fp16','int8_gs128','q4_k','iq2_xxs','tq1_0')


def effective(name,x):
    if name in ('distance','importance'):return _strength(x)
    if name.endswith('scale'):return F.softplus(x)+1e-4
    return .5*torch.tanh(x)


def bounded(name,x):
    if name in ('distance','importance'):return x.clamp(0,8)
    if name.endswith('scale'):return x.clamp_min(1e-4)
    return x.clamp(-.5,.5)


def effective_geo_step(self,coords):
    p=self._quantized_controls
    x=coords.float().clamp(-65504,65504)
    t=4*torch.tanh(x.mean(-1,keepdim=True)/4)
    z=torch.tanh(p['mean_scale'][None,:,None]*(t+p['mean_shift'][None,:,None]))
    spine=self.longitudinal(t,z,self.geo_spine)
    edge_t=self.longitudinal(t[...,None],z[...,None],self.geo_t_basis_edge)
    zx=torch.tanh(p['edge_scale'][None,:,None]*(x+p['edge_shift'][None,:,None]))[...,None]
    a,b,c,d=self.geo_x_basis_edge.unbind(1)
    edge_x=((4*d*zx+2*c)*zx+b-3*d)*zx+a-c
    distance,importance=_predicates(x)
    den=1+p['distance']*distance[...,None]+p['importance']*importance[...,None]
    return spine+(edge_t*edge_x/den).sum(-2)


def quantized(name,value,level):
    # Damping: one row per chart; warp: one row per parameter family, across charts.
    rows=value.reshape(value.shape[0],-1) if name in ('distance','importance') else value.reshape(1,-1)
    if level=='fp16':q=rows.half().float()
    elif level=='int8_gs128':q=_dequantize(rows)
    else:q,_={'q4_k':q4k,'iq2_xxs':iq2,'tq1_0':tq1}[level].roundtrip(rows)
    q=q.reshape_as(value)
    if not torch.isfinite(q).all():raise ValueError('Nonfinite decoded controls')
    return q


def summary_stats(values):
    x=torch.cat([v.flatten() for v in values])
    return dict(count=x.numel(),min=x.min().item(),max=x.max().item(),mean=x.mean().item(),std=x.std(unbiased=False).item())


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoints',type=Path,required=True)
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--fused-checkpoint',type=Path)
    args=ap.parse_args()
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(72)
    torch.set_float32_matmul_precision('highest')
    val=np.memmap(args.data_dir/'fineweb_val_000000.bin',dtype='<u2',mode='r',offset=1024)
    validation=[torch.from_numpy(np.array(val[i:i+129],dtype=np.int64)) for i in range(0,2048,128)]
    paths=[(f'original_seed_{seed}',args.checkpoints/f'ks2_{seed}.pt') for seed in (11,22,33)]
    if args.fused_checkpoint:paths.append(('fused_training_seed_11',args.fused_checkpoint))
    result=dict(protocol=dict(levels=LEVELS,scopes=SCOPES,quantization_domains=['raw','effective'],
           baseline='previously recorded checkpoint validation loss; controls not rerun',
           raw='quantize learned parameters before sigmoid/softplus/tanh transforms',
           effective='quantize bounded damping strengths, positive scales and bounded shifts after transforms; retain original safety bounds',
           layout='damping: per family, per chart 64-value rows; warp: one 7-value row per family per FFN; no per-scalar scale',
           quantizers='existing actual Q4_K/IQ2_XXS/TQ1_0 packed roundtrip wrappers, signed GS128 absmax INT8 and IEEE FP16',
           padding='INT8 final group padded to 128; Q4_K/IQ2_XXS/TQ1_0 rows padded to 256; cropped after decoding',
           importance='uniform weights for IQ2_XXS, including padding',
           computation='FP32; all polynomial coefficients and both projections remain original FP32',
           validation_tokens=2048,sequence=128,threads=8,retraining=False,calibration=False,torch=torch.__version__),
           checks=[],original_parameters={},results=[])
    def save():args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for label,path in paths:
        reference,ck=load_checkpoint(path);baseline=ck['history'][-1]['val_loss'];del ck
        original={key:{name:getattr(ff,attr).detach().clone() for name,attr in NAMES.items()} for key,ff in reference.ff.items()}
        transformed={key:{name:effective(name,v) for name,v in pp.items()} for key,pp in original.items()}
        result['original_parameters'][label]={name:dict(raw=summary_stats([pp[name] for pp in original.values()]),effective=summary_stats([pp[name] for pp in transformed.values()])) for name in NAMES}
        # Precompute all quantized tensors once; the same rounded values are used in every scope.
        values={};errors={}
        for domain,source in [('raw',original),('effective',transformed)]:
            values[domain]={};errors[domain]={}
            for level in LEVELS:
                quant={key:{name:quantized(name,v,level) for name,v in pp.items()} for key,pp in source.items()}
                if domain=='effective':
                    quant={key:{name:bounded(name,v) for name,v in pp.items()} for key,pp in quant.items()}
                values[domain][level]=quant
                errors[domain][level]={}
                for name in NAMES:
                    a=torch.cat([transformed[key][name].flatten() for key in original])
                    b=torch.cat([(effective(name,quant[key][name]) if domain=='raw' else quant[key][name]).flatten() for key in original])
                    raw_a=torch.cat([source[key][name].flatten() for key in original]);raw_b=torch.cat([quant[key][name].flatten() for key in original])
                    errors[domain][level][name]=dict(effective_rms_error=(a-b).square().mean().sqrt().item(),effective_max_error=(a-b).abs().max().item(),
                           effective_relative_rms=((a-b).square().mean()/a.square().mean().clamp_min(1e-30)).sqrt().item(),
                           changed_fraction=(raw_a!=raw_b).float().mean().item(),effective_zero_fraction=(b==0).float().mean().item())
        for domain in ('raw','effective'):
            model=copy.deepcopy(reference)
            if domain=='effective':
                for key,ff in model.ff.items():
                    ff._quantized_controls=transformed[key]
                    ff.geo_step=MethodType(effective_geo_step,ff)
                x=torch.randn(19,96)
                max_error=0.
                for key,ff in model.ff.items():
                    a=reference.ff[key](x);b=ff(x)
                    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
                    max_error=max(max_error,(a-b).abs().max().item())
                result['checks'].append(dict(checkpoint=label,effective_parameterization_max_output_error=max_error))
            for level in LEVELS:
                for scope,names in SCOPES.items():
                    for key,ff in model.ff.items():
                        if domain=='raw':
                            for name,attr in NAMES.items():getattr(ff,attr).copy_(values[domain][level][key][name] if name in names else original[key][name])
                        else:ff._quantized_controls={name:values[domain][level][key][name] if name in names else transformed[key][name] for name in NAMES}
                    ce=validate(model,validation)
                    if not math.isfinite(ce):raise ValueError('Nonfinite held-out loss')
                    result['results'].append(dict(checkpoint=label,domain=domain,level=level,scope=scope,val_ce=ce,baseline_ce=baseline,delta_ce=ce-baseline,
                                                  parameter_errors={name:errors[domain][level][name] for name in names}))
                print(label,domain,level,'combined_delta',result['results'][-1]['delta_ce'],flush=True)
                save()
            del model
        del reference
    save()


if __name__=='__main__':main()
