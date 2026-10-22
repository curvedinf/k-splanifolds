"""CPU pilot: published small patch MLP versus unmodified upstream KS2Cell."""
import argparse, csv, hashlib, json, math, os, platform, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from ks2 import KS2Cell
ROOT=Path(__file__).parent
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
P=13;D=P*P;SIGMA=25/255

def model(kind,seed):
    torch.manual_seed(seed)
    if kind=='MLP':return nn.Sequential(nn.Linear(D,511),nn.Tanh(),nn.Linear(511,511),nn.Tanh(),nn.Linear(511,D))
    return KS2Cell(D,D,num_subspaces=123,subspace_rank=8,geo_steps=2)

def read(p):return np.asarray(Image.open(p).convert('L'),dtype=np.float32)/255

def patches(paths,n,seed):
    rng=np.random.default_rng(seed);ims=[read(p) for p in paths]
    a=np.empty((n,D),np.float32)
    for i in range(n):
        im=ims[rng.integers(len(ims))];h,w=im.shape
        y=rng.integers(h-P+1);x=rng.integers(w-P+1)
        q=im[y:y+P,x:x+P]
        q=np.rot90(q,int(rng.integers(4)))
        if rng.integers(2):q=q[:,::-1]
        a[i]=q.flatten()
    return torch.from_numpy(a)

@torch.no_grad()
def validate(m,vx,vy):
    m.eval();s=0
    for x,y in zip(vx.split(128),vy.split(128)):
        s+=(m(x)-y).square().sum().item()
    m.train();return -10*math.log10(s/vy.numel()*.04)

def train(kind,seed,lr,steps,clean,vx,vy,tag):
    m=model(kind,seed);opt=torch.optim.Adam(m.parameters(),lr=lr)
    g=torch.Generator().manual_seed(seed+10000)
    best=-float('inf');best_state=None;best_step=0;elapsed=0;history=[]
    for step in range(1,steps+1):
        t=time.perf_counter()
        ids=torch.randint(len(clean),(64,),generator=g);y=clean[ids]
        x=y+torch.randn(y.shape,generator=g)*SIGMA
        x=(x-.5)/.2;y=(y-.5)/.2
        opt.zero_grad(set_to_none=True);pred=m(x);loss=(pred-y).square().mean()
        if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite {kind} {seed} {lr} {step}')
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
        elapsed+=time.perf_counter()-t
        if step==1 or step%250==0 or step==steps:
            psnr=validate(m,vx,vy)
            history.append(dict(kind=kind,seed=seed,lr=lr,step=step,train_seconds=elapsed,val_patch_psnr=psnr,loss=loss.item()))
            if psnr>best:
                best=psnr;best_step=step;best_state={k:v.detach().clone() for k,v in m.state_dict().items()}
            print(json.dumps(dict(tag=tag,**history[-1])),flush=True)
    m.load_state_dict(best_state)
    record=dict(kind=kind,seed=seed,lr=lr,steps=steps,parameters=sum(p.numel() for p in m.parameters()),train_seconds=elapsed,best_val_patch_psnr=best,best_step=best_step)
    (OUT/f'{tag}_{kind}_{seed}_{lr}_history.json').write_text(json.dumps(history,indent=2))
    if tag=='main':torch.save(dict(state_dict=best_state,record=record),OUT/f'{kind}_seed{seed}.pt')
    return m,record

@torch.no_grad()
def denoise(m,noisy):
    h,w=noisy.shape;extra_h=(-h+1)%3;extra_w=(-w+1)%3
    x=torch.from_numpy(noisy)[None,None]
    x=F.pad(x,(6,6+extra_w,6,6+extra_h),mode='reflect')
    ph,pw=x.shape[-2:];cols=F.unfold(x,P,stride=3)[0].T.contiguous()
    t=time.perf_counter()
    pred=torch.cat([m((c-.5)/.2)*.2+.5 for c in cols.split(128)])
    elapsed=time.perf_counter()-t
    pred=pred.T[None]
    img=F.fold(pred,(ph,pw),P,stride=3)
    weight=F.fold(torch.ones_like(pred),(ph,pw),P,stride=3)
    img=(img/weight)[0,0,6:6+h,6:6+w].clamp(0,1).numpy()
    return img,elapsed

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=3000);ap.add_argument('--pilot-steps',type=int,default=400);ap.add_argument('--threads',type=int,default=8);args=ap.parse_args()
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    paths=sorted((ROOT/'data/train400').glob('*'));assert len(paths)==400,len(paths)
    rng=np.random.default_rng(20261022);order=rng.permutation(len(paths));tr=[paths[i] for i in order[:360]];va=[paths[i] for i in order[360:]]
    splits={name:[str(p.relative_to(ROOT)) for p in ps] for name,ps in [('train',tr),('validation',va),('test',sorted((ROOT/'data/set12').glob('*.png')))]}
    (OUT/'splits.json').write_text(json.dumps(splits,indent=2))
    config=dict(patch_size=P,sigma=SIGMA,batch_size=64,steps=args.steps,pilot_steps=args.pilot_steps,lr_candidates=[1e-4,3e-4,1e-3],seeds=[11,22,33],train_images=360,validation_images=40,train_patch_pool=120000,validation_patches=4096,threads=args.threads,torch=torch.__version__,python=platform.python_version(),processor=platform.processor(),ks2_sha256=hashlib.sha256((ROOT/'ks2.py').read_bytes()).hexdigest(),ks2=dict(num_subspaces=123,subspace_rank=8,geo_steps=2),optimizer='Adam, gradient norm clipped to 1, constant LR',normalization='(pixel - 0.5) / 0.2',test_protocol='Set12, fixed noise seed 12345+image index, uniform overlap average stride 3, reflect padding, output clipped to [0,1]')
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    clean=patches(tr,120000,321);v=patches(va,4096,456);g=torch.Generator().manual_seed(789)
    vx=(v+torch.randn(v.shape,generator=g)*SIGMA-.5)/.2;vy=(v-.5)/.2
    pilot=[];chosen={}
    for kind in ['MLP','KS2']:
        for lr in config['lr_candidates']:
            _,r=train(kind,777,lr,args.pilot_steps,clean,vx,vy,'pilot');pilot.append(r)
        chosen[kind]=max([r for r in pilot if r['kind']==kind],key=lambda r:r['best_val_patch_psnr'])['lr']
    (OUT/'pilot.json').write_text(json.dumps(dict(runs=pilot,chosen=chosen),indent=2))
    print('SELECTED',chosen,flush=True)
    records=[]
    for seed in config['seeds']:
        for kind in ['MLP','KS2']:
            _,r=train(kind,seed,chosen[kind],args.steps,clean,vx,vy,'main');records.append(r)
            (OUT/'training.json').write_text(json.dumps(records,indent=2))
    # Only now evaluate the untouched test set.
    rows=[];visual={};test=sorted((ROOT/'data/set12').glob('*.png'))
    for seed in config['seeds']:
        for kind in ['MLP','KS2']:
            m=model(kind,seed);m.load_state_dict(torch.load(OUT/f'{kind}_seed{seed}.pt',weights_only=False)['state_dict']);m.eval()
            for i,p in enumerate(test):
                target=read(p);noise=np.random.default_rng(12345+i).normal(0,SIGMA,target.shape).astype(np.float32);noisy=target+noise
                prediction,seconds=denoise(m,noisy)
                psnr=-10*np.log10(np.mean((prediction-target)**2))
                rows.append(dict(kind=kind,seed=seed,image=p.stem,psnr=float(psnr),inference_seconds=seconds,noisy_psnr=float(-10*np.log10(np.mean(noise**2)))))
                if seed==11:visual[f'{p.stem}_{kind}']=prediction;visual[f'{p.stem}_clean']=target;visual[f'{p.stem}_noisy']=noisy
            print('TEST',seed,kind,np.mean([r['psnr'] for r in rows if r['seed']==seed and r['kind']==kind]),flush=True)
    with (OUT/'test_results.csv').open('w') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    np.savez_compressed(OUT/'visual_arrays.npz',**visual)
    summary={}
    for kind in ['MLP','KS2']:
        seed_means=[np.mean([r['psnr'] for r in rows if r['kind']==kind and r['seed']==seed]) for seed in config['seeds']]
        summary[kind]=dict(mean_psnr=float(np.mean(seed_means)),std_across_seeds=float(np.std(seed_means,ddof=1)),seed_means=seed_means,parameters=next(r['parameters'] for r in records if r['kind']==kind),mean_train_seconds=float(np.mean([r['train_seconds'] for r in records if r['kind']==kind])),mean_image_inference_seconds=float(np.mean([r['inference_seconds'] for r in rows if r['kind']==kind])))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',json.dumps(summary),flush=True)
if __name__=='__main__':main()
