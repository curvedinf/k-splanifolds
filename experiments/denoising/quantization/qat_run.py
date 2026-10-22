import json,time
from pathlib import Path
import numpy as np
import torch
import run as b
from rank_models import denoise
from qat_geometry import make_model
R=Path(__file__).parent;O=R/'qat_results';O.mkdir(exist_ok=True)
torch.set_num_threads(8);torch.set_num_interop_threads(1)
splits=json.loads((R/'results/splits.json').read_text())
clean=b.patches([R/p for p in splits['train']],120000,321);v=b.patches([R/p for p in splits['validation']],4096,456)
g=torch.Generator().manual_seed(789);vx=(v+torch.randn(v.shape,generator=g)*b.SIGMA-.5)/.2;vy=(v-.5)/.2
records=[];tests=[];diagnostics=[]
for seed in [11,22,33]:
 for kind in ['qat_combined','qat_powers']:
  m=make_model(kind,seed);opt=torch.optim.Adam(m.parameters(),lr=.0003);g=torch.Generator().manual_seed(seed+10000)
  history=[];best=-float('inf');best_state=None;elapsed=0;status='completed';failure=None
  for step in range(1,1001):
   start=time.perf_counter();ids=torch.randint(len(clean),(64,),generator=g);y=clean[ids];x=(y+torch.randn(y.shape,generator=g)*b.SIGMA-.5)/.2;target=(y-.5)/.2
   opt.zero_grad(set_to_none=True);pred=m(x);loss=(pred-target).square().mean()
   if not torch.isfinite(loss):status='nonfinite_loss';failure=step;break
   loss.backward()
   bad=[n for n,p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
   if bad:status='nonfinite_gradients';failure=step;print('FAILED',kind,seed,step,bad,flush=True);break
   torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();elapsed+=time.perf_counter()-start
   if step in [1,250,500,750,1000]:
    psnr=b.validate(m,vx,vy);h=dict(step=step,val_psnr=psnr,train_seconds=elapsed,loss=float(loss.detach()));history.append(h);print(kind,seed,json.dumps(h),flush=True)
    if psnr>best:best=psnr;best_state={k:v.detach().clone() for k,v in m.state_dict().items()}
  record=dict(kind=kind,seed=seed,status=status,failure_step=failure,completed_updates=step if status=='completed' else step-1,best_val_psnr=best if best_state is not None else None,train_seconds=elapsed,history=history)
  records.append(record);(O/'training.json').write_text(json.dumps(records,indent=2))
  if best_state is not None:m.load_state_dict(best_state);torch.save(dict(state_dict=best_state,record=record),O/f'{kind}_{seed}.pt')
  m.eval()
  with torch.no_grad():
   m.capture=True;m(vx[:128]);diagnostics.append(dict(kind=kind,seed=seed,checkpoint='best' if best_state else 'failed_training_state',stats=m.diagnostics));m.capture=False
   if status=='completed':
    for i,rel in enumerate(splits['test']):
     target=b.read(R/rel);noisy=target+np.random.default_rng(12345+i).normal(0,b.SIGMA,target.shape).astype(np.float32)
     pred,_=denoise(m,noisy);tests.append(dict(kind=kind,seed=seed,image=Path(rel).stem,psnr=float(-10*np.log10(np.mean((pred-target)**2)))))
  (O/'test.json').write_text(json.dumps(tests,indent=2));(O/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
summary={}
for kind in ['qat_combined','qat_powers']:
 rr=[r for r in records if r['kind']==kind];tt=[r for r in tests if r['kind']==kind]
 means=[np.mean([r['psnr'] for r in tt if r['seed']==seed]) for seed in [11,22,33]] if len(tt)==36 else []
 summary[kind]=dict(statuses=[r['status'] for r in rr],completed_updates=[r['completed_updates'] for r in rr],val_mean=float(np.mean([r['history'][-1]['val_psnr'] for r in rr])) if all(r['status']=='completed' for r in rr) else None,test_mean=float(np.mean(means)) if means else None,test_std=float(np.std(means,ddof=1)) if means else None)
(O/'summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',json.dumps(summary),flush=True)
