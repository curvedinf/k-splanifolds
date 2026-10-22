"""CPU numerical emulation of GPU-oriented inference precision boundaries."""
import json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from fused_inference import FrozenKS2
from rank_models import make_model,denoise
import run as bench
R=Path(__file__).parent;O=R/'quant_results';O.mkdir(exist_ok=True)
GROUPS={'longitudinal':[f'a{i}' for i in range(6)],'coordinate':[f'q{i}' for i in range(4)],'damping':['dd','dw'],'warp':['scale','shift','escale','eshift']}
class QuantKS2(FrozenKS2):
 def __init__(self,m,mode,target='all'):
  super().__init__(m);self.mode=mode;self.target=target;self.stats={}
  if mode in ['fp8','int8']:
   for group,keys in GROUPS.items():
    if target not in ['all',group]:continue
    vals=[getattr(self,k) for k in keys]
    shaped=[v.reshape(self.s,-1) for v in vals];joined=torch.cat(shaped,1)
    limit=448 if mode=='fp8' else 127
    scale=joined.abs().amax(1,keepdim=True).clamp_min(1e-12)/limit
    # Static power-of-two FP8 scales avoid extra representational ambiguity.
    if mode=='fp8':scale=2**torch.ceil(torch.log2(scale))
    if mode=='int8':
     # GS128 within each chart and coefficient family; pad only the final group.
     width=joined.shape[1];padded=F.pad(joined,(0,(-width)%128)).reshape(self.s,-1,128)
     gs=padded.abs().amax(-1,keepdim=True).clamp_min(1e-12)/127
     q=((padded/gs).round().clamp(-127,127)*gs).reshape(self.s,-1)[:,:width]
    else:
     norm=joined/scale
     q=norm.to(torch.float8_e4m3fn).float()*scale
    self.stats[group]={'relative_rmse':float(((q-joined).square().mean()/joined.square().mean().clamp_min(1e-30)).sqrt()),'new_zero_fraction':float(((q==0)&(joined!=0)).float().mean())}
    offset=0
    for k,v,sh in zip(keys,vals,shaped):
     setattr(self,k,q[:,offset:offset+sh.shape[1]].reshape(v.shape).contiguous());offset+=sh.shape[1]
 def geo(self,c):
  if self.mode!='fp16':return super().geo(c)
  # Projections remain FP32. Explicit casts emulate each FP16 geometry boundary.
  c=c.half();safe=self.target=='safe'
  total=c.float().sum(-1,keepdim=True) if safe else c.sum(-1,keepdim=True)
  t=total/self.r;near=total.abs()<1e-6
  w=torch.where(near,torch.full_like(c,1/self.r),c/torch.where(near,torch.ones_like(total),total))
  th=t.half();z=torch.tanh(self.scale.half()*(th+self.shift.half()))
  a=[getattr(self,f'a{i}').half() for i in range(6)]
  p=((a[3]*th+a[2])*th+a[1])*th+a[0]+(2*z*z-1)*a[4]+(4*z*z*z-3*z)*a[5]
  e=torch.tanh(self.escale.half()*(c+self.eshift.half()))[...,None]
  q=[getattr(self,f'q{i}').half() for i in range(4)]
  ex=((q[3]*e+q[2])*e+q[1])*e+q[0]
  if safe:
   den=1+self.dd*(c.float()-t).abs()[...,None]+self.dw*w.square()[...,None]
   edges=p[...,self.r:].reshape(-1,self.s,self.r,self.r).float()*ex.float()/den
   return c.float()+p[...,:self.r].float()+edges.sum(-2)
  den=1+self.dd.half()*(c-t).abs()[...,None]+self.dw.half()*w.square()[...,None]
  edges=p[...,self.r:].reshape(-1,self.s,self.r,self.r)*ex/den
  # torch half reduction may accumulate internally in FP32; explicitly round each addition.
  acc=torch.zeros_like(c)
  for i in range(self.r):acc=acc+edges[:,:,i,:]
  return ((c+p[...,:self.r])+acc).float()

def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument("--int8-only",action="store_true");args=ap.parse_args()
 torch.set_num_threads(8);torch.set_num_interop_threads(1)
 splits=json.loads((R/'results/splits.json').read_text())
 v=bench.patches([R/p for p in splits['validation']],4096,456);g=torch.Generator().manual_seed(789)
 vx=(v+torch.randn(v.shape,generator=g)*bench.SIGMA-.5)/.2;vy=(v-.5)/.2
 variants=[('fp32','all'),('fp16','all'),('fp16','safe')]+[(mode,target) for mode in ['fp8','int8'] for target in [*GROUPS,'all']]
 rows=[];test=[];stats=[]
 if args.int8_only:
  variants=[("int8",target) for target in [*GROUPS,"all"]]
  rows=[r for r in json.loads((O/"validation.json").read_text()) if r["mode"]!="int8"]
  test=[r for r in json.loads((O/"test.json").read_text()) if r["mode"]!="int8"]
  stats=[r for r in json.loads((O/"coefficient_stats.json").read_text()) if r["mode"]!="int8"]
 with torch.no_grad():
  for seed in [11,22,33]:
   m=make_model('R8S32',seed);m.load_state_dict(torch.load(R/f'rank_results/R8S32_seed{seed}.pt',weights_only=False)['state_dict']);m.eval()
   base=FrozenKS2(m);refs=torch.cat([base(x) for x in vx.split(128)])
   for mode,target in variants:
    model=base if mode=='fp32' else QuantKS2(m,mode,target)
    pred=torch.cat([model(x) for x in vx.split(128)]);finite=bool(torch.isfinite(pred).all())
    row=dict(seed=seed,mode=mode,target=target,finite=finite,val_psnr=float(-10*torch.log10((pred-vy).square().mean()*.04)) if finite else None,output_rmse=float((pred-refs).square().mean().sqrt()) if finite else None,max_error=float((pred-refs).abs().max()) if finite else None)
    rows.append(row);print('VAL',json.dumps(row),flush=True)
    if mode in ['fp8','int8']:stats.append(dict(seed=seed,mode=mode,target=target,groups=model.stats))
    if target=='all' or (mode=='fp16' and target=='safe'):
     for i,rel in enumerate(splits['test']):
      clean=bench.read(R/rel);noisy=clean+np.random.default_rng(12345+i).normal(0,bench.SIGMA,clean.shape).astype(np.float32)
      out,_=denoise(model,noisy)
      test.append(dict(seed=seed,mode=mode,target=target,image=Path(rel).stem,psnr=float(-10*np.log10(np.mean((out-clean)**2))) if np.isfinite(out).all() else None))
     print('TEST',seed,mode,target,flush=True)
    (O/'validation.json').write_text(json.dumps(rows,indent=2));(O/'test.json').write_text(json.dumps(test,indent=2));(O/'coefficient_stats.json').write_text(json.dumps(stats,indent=2))
 summary=[]
 for mode,target in dict.fromkeys((r["mode"],r["target"]) for r in rows):
  rr=[r for r in rows if r['mode']==mode and r['target']==target];ts=[r for r in test if r['mode']==mode and r['target']==target]
  vals=[r['val_psnr'] for r in rr]
  means=[np.mean([r['psnr'] for r in ts if r['seed']==s]) for s in [11,22,33]] if ts and all(r['psnr'] is not None for r in ts) else []
  summary.append(dict(mode=mode,target=target,val_mean=float(np.mean(vals)) if all(v is not None for v in vals) else None,val_std=float(np.std(vals,ddof=1)) if all(v is not None for v in vals) else None,test_mean=float(np.mean(means)) if means else None,test_std=float(np.std(means,ddof=1)) if means else None,max_error=max([r['max_error'] or 0 for r in rr])))
 (O/'summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',json.dumps(summary),flush=True)
if __name__=='__main__':main()
