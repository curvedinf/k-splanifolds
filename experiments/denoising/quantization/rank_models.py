"""Whole multivariate KS2 units: packed eager training, exact fused inference."""
import time,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from parallel_models import ParallelKS2
from fast_models import HornerKS2
import torch._inductor.config as inductor_config
# Large OpenMP fused kernels were slower on this workload; dense GEMMs still use 8 threads.
inductor_config.cpp.threads=1
CONFIGS={'R8S32':(8,32),'R8S64':(8,64),'R16S50':(16,50),'R32S18':(32,18)}
CLASSES=CONFIGS
DEPLOY=[]
def make_model(name,seed):
    torch.manual_seed(seed);rank,spaces=CONFIGS[name]
    return ParallelKS2(spaces,1,'sum',r=rank)

def prepare_inference(trained,name,seed):
    rank,spaces=CONFIGS[name];h=HornerKS2(spaces,rank)
    h.load_state_dict(trained.state_dict());h.eval()
    f=torch.compile(h,fullgraph=True,dynamic=False)
    g=torch.Generator().manual_seed(8765)
    x=torch.randn(128,169,generator=g)
    with torch.no_grad():
        reference=trained(x);t=time.perf_counter();actual=f(x);warm=time.perf_counter()-t
        torch.testing.assert_close(actual,reference,atol=1e-5,rtol=1e-4)
        for _ in range(3):f(x)
    DEPLOY.append(dict(model=name,seed=seed,first_call_seconds=warm,max_output_error=float((actual-reference).abs().max())))
    print('DEPLOY',json.dumps(DEPLOY[-1]),flush=True)
    return f

@torch.no_grad()
def denoise(m,noisy):
    h,w=noisy.shape;eh=(-h+1)%3;ew=(-w+1)%3
    x=F.pad(torch.from_numpy(noisy)[None,None],(6,6+ew,6,6+eh),mode='reflect')
    ph,pw=x.shape[-2:];cols=F.unfold(x,13,stride=3)[0].T.contiguous()
    pred=[];t=time.perf_counter()
    for c in cols.split(128):
        n=len(c);inp=(c-.5)/.2
        if n!=128:inp=F.pad(inp,(0,0,0,128-n))
        pred.append((m(inp)*.2+.5)[:n])
    pred=torch.cat(pred).T[None];seconds=time.perf_counter()-t
    img=F.fold(pred,(ph,pw),13,stride=3)
    weight=F.fold(torch.ones_like(pred),(ph,pw),13,stride=3)
    img=(img/weight)[0,0,6:6+h,6:6+w].clamp(0,1).numpy()
    return img,seconds
