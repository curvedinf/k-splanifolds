"""Check structural controls, initial function, finite gradients and causality."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
import run
from gpu_recurrence import recurrence
from study_models_v19 import build_model,make_optimizer,parameter_breakdown
from side_stream import BaseModel

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--stride',type=int,default=4);a=p.parse_args()
 torch.set_num_threads(2);torch.manual_seed(591);tokens=torch.randint(256,(2,8));reference=run.recurrence
 jobs=list({json.dumps(j['config'],sort_keys=True):j for j in json.loads(a.manifest.read_text())['jobs']}.values())
 for job in jobs[a.shard::a.stride]:
  cfg=job['config'];run.recurrence=reference
  unfused=dict(cfg,fused_geometry=False,fused_partial=False,fused_general=False);model=build_model(11,unfused);width=cfg.get('width',96)
  reference_cfg={k:v for k,v in unfused.items() if k not in ('geometry_steps','geometry_step_scale','geometry_shuffle','geometry_trainable','subspace_rank','geometry_basis','geometry_input_norm','geometry_placement','geometry_calibration','frozen_geometry','geometry_equation','geometry_input_sigmoid','spine_input_scale','mixed_ranks','mlp_features')}
  baseline=build_model(11,dict(reference_cfg,family='mlp'))
  with torch.no_grad():torch.testing.assert_close(model(tokens),baseline(tokens),atol=0,rtol=0)
  for i,block in enumerate(model.blocks):
   for name,value in block.attention.state_dict().items():assert torch.equal(value,baseline.blocks[i].attention.state_dict()[name])
  del baseline
  if cfg.get('fused_geometry') or cfg.get('fused_partial') or cfg.get('fused_general'):
   if cfg.get('mixed_ranks'):
    eager=model.blocks[0].ffn.cuda();fast=build_model(11,cfg).blocks[0].ffn.cuda()
    with torch.no_grad():eager.up.weight.normal_(0,.02);fast.up.weight.copy_(eager.up.weight)
    x=torch.randn(2,8,width,device='cuda',requires_grad=True);y=x.detach().clone().requires_grad_(True)
    left=eager(x);right=fast(y);torch.testing.assert_close(left,right,atol=2e-5,rtol=2e-5)
    weight=torch.randn_like(left);(left*weight).sum().backward();(right*weight).sum().backward()
    torch.testing.assert_close(x.grad,y.grad,atol=2e-5,rtol=2e-5)
    for (name,ep),(other,fp) in zip(eager.named_parameters(),fast.named_parameters()):
     assert name==other
     torch.testing.assert_close(ep.grad,fp.grad,atol=2e-5,rtol=2e-5,msg=lambda message:name+' '+message)
    del eager,fast,x,y,left,right,weight
   model=build_model(11,cfg)
  model=model.cuda();run.recurrence=recurrence;opt=make_optimizer(model)
  for _ in range(4):
   opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(tokens.cuda()).flatten(0,1),tokens.roll(-1,1).cuda().flatten());loss.backward()
   torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
  for name,param in model.named_parameters():
   if name=='blocks.0.attention.value_bias':assert param.grad is None
   else:assert param.grad is not None and torch.isfinite(param.grad).all(),name
  changed=tokens.clone();changed[:,5:]=(changed[:,5:]+1)%256
  with torch.no_grad():torch.testing.assert_close(model(tokens.cuda())[:,:5],model(changed.cuda())[:,:5],atol=2e-6,rtol=2e-6)
  print(json.dumps(dict(status='PASS',config=cfg,parameters=parameter_breakdown(model))),flush=True)
  del model,opt
if __name__=='__main__':main()
