"""Compact stabilized KS2 denoising with all-weight padded GS128 ternary QAT."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
from types import MethodType

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch import nn
from experiments.denoising import run as b
from experiments.common.ks2_stabilized import KS2StabilizedCell
from experiments.common.ks2_cpu.training import fuse_training
from experiments.nanogpt_ks2.train_qat import forward,weights,export


def model(seed,threads,qat=True):
    torch.manual_seed(seed)
    m=KS2StabilizedCell(169,169,32,8,1)
    # Preserve the historical packed-model initialization draw order.
    torch.manual_seed(seed)
    nn.Linear(169,256,bias=False);nn.Linear(256,169,bias=False)
    t=torch.randn(32,6,72)*.05;x=torch.randn(32,8,4,8)*.05
    with torch.no_grad():
        m.geo_spine.copy_(t[:,:,:8]);m.geo_t_basis_edge.copy_(t[:,:,8:].reshape_as(m.geo_t_basis_edge))
        m.geo_x_basis_edge.copy_(x.permute(0,2,1,3))
    fuse_training(m,threads)
    if qat:m.forward=MethodType(forward,m)
    return m


@torch.no_grad()
def packed(m):
    expected=weights(m)
    wrapper=nn.Module();wrapper.ff=nn.ModuleDict({'cell':m})
    size=export(wrapper)
    for name,p in m.named_parameters():torch.testing.assert_close(p,expected[name],rtol=0,atol=0)
    fuse_training(m,m._backward_threads)
    return size


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=3000)
    ap.add_argument('--seeds',type=int,nargs='+',default=[11,22,33])
    ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--lr',type=float,default=.0003)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest=b.dataset(a.data_dir);b.save(a.out/'data_manifest.json',manifest)
    paths=sorted((a.data_dir/'train400').glob('test_*.png'))
    order=np.random.default_rng(20261022).permutation(400)
    tr,va=[paths[i] for i in order[:360]],[paths[i] for i in order[360:]]
    test=[a.data_dir/'set12'/f'{i:02d}.png' for i in range(1,13)]
    b.save(a.out/'splits.json',{k:[str(p.relative_to(a.data_dir)) for p in ps] for k,ps in [('train',tr),('validation',va),('test',test)]})
    clean=b.patches(tr,120000,321);v=b.patches(va,4096,456)
    vx=(v+torch.randn(v.shape,generator=torch.Generator().manual_seed(789))*b.SIGMA-.5)/.2;vy=(v-.5)/.2
    checkpoints=sorted(set([min(1000,a.steps),a.steps]))
    b.save(a.out/'config.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        protocol='Train400/Set12 original split/noise/normalization; batch64; Adam constant LR .0003, gradient clip1; from scratch; no control reruns',
        model='32 charts, rank8, one stabilized geometry step; 112768 parameters; historical packed initialization',
        quantization='All learned weights: GS128 ternary, FP16 absmax scales, row padding; identity STE; FP32 masters/activations/backward; fused analytic backward',
        validation='4096 fixed patches; every250 steps and step1; pixel MSE',test='Set12 fixed checkpoints at1000/final; no best-checkpoint substitution; original stride3 reconstruction',
        source_hashes={str(p):b.digest(p.read_bytes()) for p in [Path(__file__),Path(b.__file__),Path(sys.modules[forward.__module__].__file__)]},
        dataset_sha256=b.DATA_DIGEST,torch=torch.__version__))
    records=[]
    for seed in a.seeds:
        m=model(seed,a.threads);opt=torch.optim.Adam(m.parameters(),lr=a.lr)
        g=torch.Generator().manual_seed(seed+10000);history=[];start=time.perf_counter()
        for step in range(1,a.steps+1):
            y=clean[torch.randint(len(clean),(64,),generator=g)]
            x=(y+torch.randn(y.shape,generator=g)*b.SIGMA-.5)/.2
            opt.zero_grad(set_to_none=True);loss=(m(x)-(y-.5)/.2).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
            if step==1 or step%250==0 or step==a.steps:
                val=b.validate(m,vx,vy)
                row=dict(step=step,train_mse_pixel=loss.item()*.04,val_mse_pixel=val,val_psnr=-10*math.log10(val),seconds=time.perf_counter()-start)
                history.append(row);print(seed,json.dumps(row),flush=True)
                b.save(a.out/f'history_{seed}.json',history)
            if step in checkpoints:torch.save(m.state_dict(),a.out/f'ks2_{seed}_{step}.pt')
        records.append(dict(seed=seed,history=history,seconds=time.perf_counter()-start))
        b.save(a.out/'training.json',records)
    tests=[];checks=[]
    for seed in a.seeds:
        for step in checkpoints:
            m=model(seed,a.threads);m.load_state_dict(torch.load(a.out/f'ks2_{seed}_{step}.pt',weights_only=True))
            fake=b.validate(m,vx,vy);size=packed(m);actual=b.validate(m,vx,vy)
            assert abs(fake-actual)<1e-8
            checks.append(dict(seed=seed,step=step,qat_val_mse=fake,packed_val_mse=actual,packed_bytes=size))
            m.eval()
            for i,p in enumerate(test):
                target=b.read(p);noise=np.random.default_rng(12345+i).normal(0,b.SIGMA,target.shape).astype(np.float32)
                with torch.no_grad():pred,_=b.denoise(m,target+noise)
                mse=float(np.mean((pred-target)**2))
                tests.append(dict(seed=seed,step=step,image=p.name,mse_pixel=mse,psnr=-10*math.log10(mse)))
            b.save(a.out/'test.json',tests);b.save(a.out/'export_checks.json',checks)
            print('TEST',seed,step,np.mean([x['psnr'] for x in tests if x['seed']==seed and x['step']==step]),flush=True)
    summary={}
    for step in checkpoints:
        rows=[x for x in tests if x['step']==step]
        means=[float(np.mean([x['psnr'] for x in rows if x['seed']==s])) for s in a.seeds]
        summary[str(step)]=dict(test_mse_pixel=float(np.mean([x['mse_pixel'] for x in rows])),test_psnr=float(np.mean(means)),
            seed_psnr=means,seed_psnr_std=float(np.std(means,ddof=1)) if len(means)>1 else 0.,
            val_mse_pixel=float(np.mean([x['packed_val_mse'] for x in checks if x['step']==step])),packed_bytes=checks[0]['packed_bytes'])
    b.save(a.out/'summary.json',summary);print('SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__':main()
