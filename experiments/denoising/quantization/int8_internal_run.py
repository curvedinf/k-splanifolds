from pathlib import Path
import json,numpy as np,torch
import run as b
from rank_models import denoise
from int8_geometry import IntegerGeometry
R=Path(__file__).parent;O=R/'int8_internal_results';O.mkdir(exist_ok=True)
torch.set_num_threads(8);torch.set_num_interop_threads(1)
splits=json.loads((R/'results/splits.json').read_text())
cal=b.patches([R/p for p in splits['train']],1024,9876);g=torch.Generator().manual_seed(9877);cx=(cal+torch.randn(cal.shape,generator=g)*b.SIGMA-.5)/.2
v=b.patches([R/p for p in splits['validation']],4096,456);g=torch.Generator().manual_seed(789);vx=(v+torch.randn(v.shape,generator=g)*b.SIGMA-.5)/.2;vy=(v-.5)/.2
modes=['reference','contracts','edges','combined','powers_all'];rows=[];tests=[];diagnostics=[]
with torch.no_grad():
 for seed in [11,22,33]:
  state=torch.load(R/f'cubic_results/bounded_{seed}.pt',weights_only=False)['state_dict']
  m=IntegerGeometry();m.load_state_dict(state);m.calibrating=True
  for x in cx.split(128):m(x)
  m.calibrating=False;m.prepare_weights();ref=None
  for mode in modes:
   m.mode=mode;m.clips={};m.count_clips=True
   pred=torch.cat([m(x) for x in vx.split(128)]);m.count_clips=False
   assert torch.isfinite(pred).all()
   if ref is None:ref=pred
   row=dict(seed=seed,mode=mode,val_psnr=float(-10*torch.log10((pred-vy).square().mean()*.04)),rmse=float((pred-ref).square().mean().sqrt()),max_error=float((pred-ref).abs().max()))
   rows.append(row);diagnostics.append(dict(seed=seed,mode=mode,clipping=m.clips));print('VAL',json.dumps(row),flush=True)
   for i,rel in enumerate(splits['test']):
    clean=b.read(R/rel);noisy=clean+np.random.default_rng(12345+i).normal(0,b.SIGMA,clean.shape).astype(np.float32);out,_=denoise(m,noisy)
    assert np.isfinite(out).all();tests.append(dict(seed=seed,mode=mode,image=Path(rel).stem,psnr=float(-10*np.log10(np.mean((out-clean)**2)))))
   print('TEST',seed,mode,flush=True)
   (O/'validation.json').write_text(json.dumps(rows,indent=2));(O/'test.json').write_text(json.dumps(tests,indent=2));(O/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
summary={}
for mode in modes:
 rr=[r for r in rows if r['mode']==mode];tt=[r for r in tests if r['mode']==mode];means=[np.mean([r['psnr'] for r in tt if r['seed']==seed]) for seed in [11,22,33]]
 summary[mode]=dict(val_psnr=float(np.mean([r['val_psnr'] for r in rr])),test_psnr=float(np.mean(means)),test_std=float(np.std(means,ddof=1)),max_error=max(r['max_error'] for r in rr))
(O/'summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',json.dumps(summary),flush=True)
