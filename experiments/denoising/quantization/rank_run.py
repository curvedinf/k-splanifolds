"""Run only the four new variants, exactly 1000 steps per seed, no LR trials."""
import csv,json,hashlib,time,platform
from pathlib import Path
import numpy as np
import torch
import run as bench
from rank_models import CONFIGS,CLASSES,make_model,prepare_inference,denoise,DEPLOY
ROOT=Path(__file__).parent;OUT=ROOT/'rank_results';OUT.mkdir(exist_ok=True)

def main():
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    bench.OUT=OUT;bench.model=make_model;bench.denoise=denoise
    splits=json.loads((ROOT/'results/splits.json').read_text())
    clean=bench.patches([ROOT/p for p in splits['train']],120000,321)
    v=bench.patches([ROOT/p for p in splits['validation']],4096,456)
    g=torch.Generator().manual_seed(789)
    vx=(v+torch.randn(v.shape,generator=g)*bench.SIGMA-.5)/.2;vy=(v-.5)/.2
    config=dict(steps=1000,seeds=[11,22,33],lr=.0003,batch_size=64,threads=8,models=list(CLASSES),lr_policy='Reuse original KS2 chosen LR, no new tuning trials',torch=torch.__version__,source_sha256=hashlib.sha256((ROOT/'rank_models.py').read_bytes()).hexdigest(),benchmark='Same splits, clean patch pool, augmentation, fresh per-step noise, optimizer, clipping, normalization and test reconstruction as prior run')
    config.update(architectures={k:dict(rank=r,subspaces=n) for k,(r,n) in CONFIGS.items()},training_backend='packed eager',inference_backend='compiled exact Horner, fixed batch128',inductor_cpp_threads=1)
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    records=[]
    for seed in config['seeds']:
        for kind in CLASSES:
            _,r=bench.train(kind,seed,config['lr'],1000,clean,vx,vy,'main');records.append(r)
            (OUT/'training.json').write_text(json.dumps(records,indent=2))
    rows=[];visual={}
    for seed in config['seeds']:
        for kind in CLASSES:
            m=make_model(kind,seed)
            m.load_state_dict(torch.load(OUT/f'{kind}_seed{seed}.pt',weights_only=False)['state_dict']);m.eval()
            m=prepare_inference(m,kind,seed)
            (OUT/'deployment.json').write_text(json.dumps(DEPLOY,indent=2))
            for i,rel in enumerate(splits['test']):
                p=ROOT/rel;target=bench.read(p)
                noise=np.random.default_rng(12345+i).normal(0,bench.SIGMA,target.shape).astype(np.float32)
                noisy=target+noise;pred,seconds=bench.denoise(m,noisy)
                psnr=float(-10*np.log10(np.mean((pred-target)**2)))
                rows.append(dict(kind=kind,seed=seed,image=p.stem,psnr=psnr,inference_seconds=seconds))
                if seed==11:
                    visual[f'{p.stem}_{kind}']=pred
                    visual[f'{p.stem}_clean']=target;visual[f'{p.stem}_noisy']=noisy
            print('TEST',seed,kind,np.mean([r['psnr'] for r in rows if r['kind']==kind and r['seed']==seed]),flush=True)
    with (OUT/'test_results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(OUT/'visual_arrays.npz',**visual)
    summary={}
    for k in CLASSES:
        means=[float(np.mean([r['psnr'] for r in rows if r['kind']==k and r['seed']==seed])) for seed in config['seeds']]
        rr=[r for r in records if r['kind']==k]
        hs=[json.loads(p.read_text()) for p in sorted(OUT.glob(f'main_{k}_*_history.json'))]
        summary[k]=dict(parameters=rr[0]['parameters'],mean_psnr=float(np.mean(means)),std_across_seeds=float(np.std(means,ddof=1)),seed_means=means,mean_train_seconds=float(np.mean([r['train_seconds'] for r in rr])),mean_image_inference_seconds=float(np.mean([r['inference_seconds'] for r in rows if r['kind']==k])),val_psnr_at_1000=float(np.mean([h[-1]['val_patch_psnr'] for h in hs])),std_val_psnr_at_1000=float(np.std([h[-1]['val_patch_psnr'] for h in hs],ddof=1)))
    # Reuse old logs; never train or evaluate the controls again.
    old=json.loads((ROOT/'results/summary.json').read_text());refs={}
    for k in ['MLP','KS2']:
        hs=[json.loads(p.read_text()) for p in sorted((ROOT/'results').glob(f'main_{k}_*_history.json'))]
        rs=[next(r for r in h if r['step']==1000) for h in hs]
        refs[k]=dict(old[k],test_steps=3000,val_steps=1000,val_psnr_at_1000=float(np.mean([r['val_patch_psnr'] for r in rs])),std_val_psnr_at_1000=float(np.std([r['val_patch_psnr'] for r in rs],ddof=1)),train_seconds_at_1000=float(np.mean([r['train_seconds'] for r in rs])))
    (OUT/'summary.json').write_text(json.dumps(dict(new=summary,previous_references=refs),indent=2))
    print('SUMMARY',json.dumps(summary),flush=True)
if __name__=='__main__':main()
