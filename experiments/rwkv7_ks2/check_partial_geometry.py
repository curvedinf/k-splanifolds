"""Random, edge-case, trained-state, and capture checks for reduced KS2 kernels."""
import argparse,copy,json,time,hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
import run
from gpu_recurrence import recurrence
from study_models_v3 import TermCell,build_model
from experiments.common.ks2_stabilized import KS2StabilizedCell
from ks2_partial_gpu import install_partial

def main():
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['undamped','spine'],required=True);p.add_argument('--trained',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(2);torch.manual_seed(844);cases=[]
 for n,s,scale in [(1,1,1.),(17,3,.01),(256,33,1.),(33,4,10.),(9,2,0.),(5,2,1e5)]:
  ref=TermCell(KS2StabilizedCell(8,8,num_subspaces=s,geo_steps=1),dict(geometry_terms=a.kind)).cuda()
  with torch.no_grad():
   for name,p in ref.named_parameters(recurse=False):
    if 'raw_' in name:p.add_(torch.randn_like(p)*.4)
  fast=install_partial(copy.deepcopy(ref));x=(torch.randn(n,s,8,device='cuda')*scale).requires_grad_();xx=x.detach().clone().requires_grad_();u=torch.randn_like(x)
  y,yy=ref.geo_step(x),fast.geo_step(xx);y.backward(u);yy.backward(u);torch.testing.assert_close(yy,y,atol=3e-5,rtol=3e-4)
  errors={}
  for name,g,gg in [('input',x.grad,xx.grad)]+[(name,p.grad,getattr(fast,name).grad) for name,p in ref.named_parameters(recurse=False)]:
   torch.testing.assert_close(gg,g,atol=3e-4,rtol=2e-3);errors[name]=float((gg-g).norm()/g.norm().clamp_min(1e-9))
  cases.append(dict(n=n,charts=s,scale=scale,output_max_error=float((yy-y).detach().abs().max()),gradient_relative_l2=errors))
  print(json.dumps(cases[-1]),flush=True)
 r=json.loads((a.trained/'results.json').read_text());cfg=r['study_config'];ref=build_model(r['arguments']['seed'],cfg)
 ref.load_state_dict(torch.load(a.trained/'checkpoint.pt',map_location='cpu',weights_only=True)['model']);fast=copy.deepcopy(ref)
 for b in fast.blocks:install_partial(b.ffn)
 ref,fast=ref.cuda(),fast.cuda();run.recurrence=recurrence
 _,val,_=run.prepare_data(Path(r['arguments']['data']),a.trained);batch=val[torch.tensor([0,128,256,384])[:,None]+torch.arange(65)].cuda()
 outputs=[]
 for m in (ref,fast):
  m.zero_grad(set_to_none=True);y=m(batch[:,:-1]);loss=F.cross_entropy(y.flatten(0,1),batch[:,1:].flatten());loss.backward();outputs.append((y.detach(),loss.detach()))
 torch.testing.assert_close(outputs[0][0],outputs[1][0],atol=1e-3,rtol=1e-3);torch.testing.assert_close(outputs[0][1],outputs[1][1],atol=1e-5,rtol=1e-6)
 errors={}
 for (name,p1),(name2,p2) in zip(ref.named_parameters(),fast.named_parameters()):
  assert name==name2
  if p1.grad is None:assert p2.grad is None;continue
  errors[name]=float((p1.grad-p2.grad).norm()/p1.grad.norm().clamp_min(1e-8));assert errors[name]<.01,(name,errors[name])
 trained=dict(logit_max_error=float((outputs[0][0]-outputs[1][0]).abs().max()),loss_reference=float(outputs[0][1]),loss_fused=float(outputs[1][1]),gradient_max_relative_l2=max(errors.values()))
 timings={}
 for label,m in [('reference',ref),('fused',fast)]:
  cell=copy.deepcopy(m.blocks[0].ffn);x=torch.randn(256,cell.num_subspaces,8,device='cuda',requires_grad=True)
  def update():
   cell.zero_grad(set_to_none=True);x.grad=None;y=cell.geo_step(x);y.square().mean().backward();return y
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(3):update()
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();expected=cell.geo_step(x).detach().clone();graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):y=update()
  graph.replay();torch.testing.assert_close(y,expected);begin=time.perf_counter()
  for _ in range(300):graph.replay()
  torch.cuda.synchronize();timings[label]=(time.perf_counter()-begin)*1000/300
 result=dict(status='PASS',kind=a.kind,cases=cases,trained=trained,geometry_forward_backward_ms=timings,source_sha256=hashlib.sha256(Path(__file__).with_name('ks2_partial_gpu.py').read_bytes()).hexdigest())
 a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status='PASS',trained=trained,timings=timings)),flush=True)
if __name__=='__main__':main()
