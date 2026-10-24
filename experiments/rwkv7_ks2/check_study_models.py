"""Validate study interventions, common initial function, gradients and causality."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
import run
from gpu_recurrence import recurrence
from study_models import build_model,make_optimizer,parameter_breakdown,MainReadout


def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--shard',type=int,default=0);args=p.parse_args()
 torch.set_num_threads(2)
 configs=[j['config'] for j in json.loads(args.manifest.read_text())['jobs']][args.shard::4]
 tokens=torch.randint(256,(2,8));baseline=build_model(11,{'name':'baseline'})
 with torch.no_grad():initial=baseline(tokens)
 reference_recurrence=run.recurrence;report=[]
 for cfg in configs:
  run.recurrence=reference_recurrence
  model=build_model(11,cfg)
  with torch.no_grad():torch.testing.assert_close(model(tokens),initial,atol=0,rtol=0)
  for i,b in enumerate(model.blocks):
   assert b.input_width==2144+192*i and b.output_width==2144
   for n,v in b.attention.state_dict().items():assert torch.equal(v,baseline.blocks[i].attention.state_dict()[n])
   if isinstance(b.ffn.up,MainReadout):
    x=torch.randn(2,3,b.ffn.up.matrix.shape[1])
    with torch.no_grad():b.ffn.up.weight.fill_(.03)
    torch.testing.assert_close(b.ffn.up(x),F.linear(x,b.ffn.up.matrix)*b.ffn.up.weight,atol=2e-6,rtol=2e-5)
    with torch.no_grad():b.ffn.up.weight.zero_()
  model=model.cuda();run.recurrence=recurrence;opt=make_optimizer(model,True)
  for _ in range(4):
   opt.zero_grad(set_to_none=True)
   loss=F.cross_entropy(model(tokens.cuda()).flatten(0,1),tokens.roll(-1,1).cuda().flatten())
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
  for name,param in model.named_parameters():
   if name=='blocks.0.attention.value_bias':
    assert param.grad is None
   else:assert param.grad is not None and torch.isfinite(param.grad).all(),name
  assert all(b.ffn.up.weight.grad[:96].abs().max()>0 for b in model.blocks)
  changed=tokens.clone();changed[:,5:]=(changed[:,5:]+1)%256
  with torch.no_grad():torch.testing.assert_close(model(tokens.cuda())[:,:5],model(changed.cuda())[:,:5],atol=2e-6,rtol=2e-6)
  report.append(dict(config=cfg,parameters=parameter_breakdown(model)))
  print(json.dumps(dict(event='checked',name=cfg['name'],parameters=parameter_breakdown(model)['total'])),flush=True)
  del model,opt
 print(json.dumps(dict(status='PASS',shard=args.shard,arms=report),indent=2))


if __name__=='__main__':main()
