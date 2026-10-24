"""Numerical, masking, gradient, graph, and timing tests for fused KS2 geometry."""
import copy,json,time,argparse
from pathlib import Path
import torch
import run
from experiments.common.ks2_stabilized import KS2StabilizedCell
from ks2_geometry_gpu import install,NAMES


def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(2);torch.manual_seed(811)
 rows=[]
 for n,s,scale in [(1,1,1.),(17,3,.01),(256,64,1.),(33,4,10.),(9,2,0.),(5,2,1e5)]:
  ref=KS2StabilizedCell(8,8,num_subspaces=s,geo_steps=1).cuda()
  fused=install(copy.deepcopy(ref));x=(torch.randn(n,s,8,device='cuda')*scale).requires_grad_();xx=x.detach().clone().requires_grad_()
  u=torch.randn_like(x);y=ref.geo_step(x);yy=fused.geo_step(xx)
  y.backward(u);yy.backward(u)
  torch.testing.assert_close(yy,y,atol=3e-5,rtol=3e-4)
  errors={}
  for name,g,gg in [('input',x.grad,xx.grad)]+[(name,getattr(ref,name).grad,getattr(fused,name).grad) for name in NAMES]:
   torch.testing.assert_close(gg,g,atol=3e-4,rtol=2e-3)
   errors[name]=dict(max_abs=float((gg-g).abs().max()),relative_l2=float((gg-g).norm()/g.norm().clamp_min(1e-10)))
  rows.append(dict(n=n,charts=s,scale=scale,output_error=float((yy-y).abs().max()),gradients=errors));print(json.dumps(rows[-1]),flush=True)
 # Benchmark captured forward/backward without optimizer and check captured output.
 timings={}
 for label,fuse in [('reference',False),('fused',True)]:
  cell=KS2StabilizedCell(8,8,num_subspaces=64,geo_steps=1).cuda()
  if fuse:install(cell)
  x=torch.randn(256,64,8,device='cuda',requires_grad=True)
  def update():
   for p in cell.parameters():p.grad=None
   x.grad=None
   y=cell.geo_step(x);y.square().mean().backward();return y
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(3):update()
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
  expected=cell.geo_step(x).detach().clone()
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):y=update()
  graph.replay();torch.testing.assert_close(y,expected)
  begin=time.perf_counter()
  for _ in range(300):graph.replay()
  torch.cuda.synchronize();timings[label]=(time.perf_counter()-begin)*1000/300
 a.out.write_text(json.dumps(dict(status='PASS',cases=rows,geometry_forward_backward_ms=timings),indent=2)+'\n')
 print(json.dumps(dict(status='PASS',timings=timings)),flush=True)
if __name__=='__main__':main()
