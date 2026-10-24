"""Fused geometry comparison on a trained history model, all parameter gradients."""
import copy,json,argparse
from pathlib import Path
import torch
from torch.nn import functional as F
import run
from gpu_recurrence import recurrence
from study_models_v2 import build_model
from ks2_geometry_gpu import install,NAMES

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
 run.recurrence=recurrence
 ref=build_model(11,{'name':'history'});ref.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True)['model'])
 fused=copy.deepcopy(ref)
 for b in fused.blocks:install(b.ffn)
 ref=ref.cuda();fused=fused.cuda();torch.manual_seed(822)
 # First compare every trained geometry parameter directly, before recurrence can amplify rounding.
 direct=[]
 for i,(b1,b2) in enumerate(zip(ref.blocks,fused.blocks)):
  c1,c2=b1.ffn,b2.ffn;u=torch.randn(17,64,8,device='cuda');x1=torch.randn_like(u,requires_grad=True);x2=x1.detach().clone().requires_grad_()
  y1,y2=c1.geo_step(x1),c2.geo_step(x2);y1.backward(u);y2.backward(u)
  torch.testing.assert_close(y1,y2,atol=1e-5,rtol=1e-4)
  err={}
  for name,g1,g2 in [('input',x1.grad,x2.grad)]+[(name,getattr(c1,name).grad,getattr(c2,name).grad) for name in NAMES]:
   torch.testing.assert_close(g1,g2,atol=2e-4,rtol=1e-3);err[name]=float((g1-g2).norm()/g1.norm().clamp_min(1e-9))
  direct.append(dict(layer=i,output_max_error=float((y1-y2).abs().max()),relative_l2_gradient_errors=err))
 _,val,_=run.prepare_data(Path(__file__).parent/'gpu_runs/data',a.out.parent)
 batch=val[torch.tensor([0,128,256,384])[:,None]+torch.arange(65)].cuda();x,target=batch[:,:-1],batch[:,1:]
 rows=[]
 for step in range(3):
  fused.load_state_dict(ref.state_dict())
  yy=[]
  for m in (ref,fused):
   m.zero_grad(set_to_none=True);y=m(x);loss=F.cross_entropy(y.flatten(0,1),target.flatten());loss.backward();yy.append((y.detach(),loss.detach()))
  torch.testing.assert_close(yy[1][0],yy[0][0],atol=1e-3,rtol=1e-3)
  torch.testing.assert_close(yy[1][1],yy[0][1],atol=1e-5,rtol=1e-6)
  grad_errors={}
  for (name,p1),(name2,p2) in zip(ref.named_parameters(),fused.named_parameters()):
   assert name==name2
   if p1.grad is None:assert p2.grad is None;continue
   grad_errors[name]=float((p1.grad-p2.grad).norm()/p1.grad.norm().clamp_min(1e-8))
   assert grad_errors[name]<.01,(name,grad_errors[name])
  rows.append(dict(step=step,loss_reference=float(yy[0][1]),loss_fused=float(yy[1][1]),logit_max_error=float((yy[0][0]-yy[1][0]).abs().max()),max_gradient_relative_l2=max(grad_errors.values())))
  # Match the actual norm-clipped update; compare functions at a common parameter state each time.
  torch.nn.utils.clip_grad_norm_(ref.parameters(),1.,error_if_nonfinite=True)
  with torch.no_grad():
   for p in ref.parameters():
    if p.grad is not None:p.add_(p.grad,alpha=-1e-4)
 a.out.write_text(json.dumps(dict(status='PASS',trained_geometry=direct,steps=rows,note='Common parameter states; actual validation bytes; clipped updates. Arithmetic is close, not bit-identical. Earlier random-byte un-clipped trajectory check amplified small rounding differences.'),indent=2)+'\n');print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
