"""GPU kernel values, masked sizes, backward, and captured training checks."""
import argparse,json,random,statistics
from pathlib import Path
import torch
from torch.nn import functional as F
from spread_projection import make_codes
from structured_codes import compact_codes
from structured_codes_gpu import Mux,Demux,mux,transpose


def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(819)
 rows=[]
 for width in (96,2144,2336,2528,2720):
  charts=64;codes,_,_=make_codes(width,charts);matrix=codes.flatten(0,1).cuda();scale=float(matrix[0,0].abs())
  perm,inv,signs=[z.cuda() for z in compact_codes(width,charts)]
  for n in (3,17,256):
   x=torch.randn(n,width,device='cuda',requires_grad=True);u=torch.randn(n,charts*8,device='cuda',requires_grad=True)
   y=Mux.apply(x,perm,inv,signs,scale);target=F.linear(x,matrix);torch.testing.assert_close(y,target,atol=1e-5,rtol=5e-5)
   g=torch.autograd.grad((y*u).sum(),x)[0];gt=F.linear(u,matrix.T);torch.testing.assert_close(g,gt,atol=1e-5,rtol=5e-5)
   z=Demux.apply(u,perm,inv,signs,scale);torch.testing.assert_close(z,gt,atol=1e-5,rtol=5e-5)
   h=torch.autograd.grad((z*x).sum(),u)[0];torch.testing.assert_close(h,target,atol=1e-5,rtol=5e-5)
   rows.append(dict(width=width,batch=n,mux_error=(y-target).abs().max().item(),transpose_error=(z-gt).abs().max().item(),gradient_error=(g-gt).abs().max().item()))
  x=torch.randn(17,width,device='cuda',requires_grad=True)
  def update():
   x.grad=None;y=Mux.apply(x,perm,inv,signs,scale);z=Demux.apply(y,perm,inv,signs,scale);loss=z.square().mean();loss.backward();return loss
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(3):update()
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):loss=update()
  for _ in range(3):graph.replay()
  torch.cuda.synchronize();assert torch.isfinite(x.grad).all() and torch.isfinite(loss);del graph
  print(json.dumps(dict(width=width,status='PASS')),flush=True)
 a.out.write_text(json.dumps(dict(status='PASS',rows=rows),indent=2)+'\n')
 # Measure each fixed operator and its transpose separately, on the same GPU.
 import triton.testing
 cases=[]
 for width in (96,2144,2336,2528,2720):
  codes,_,_=make_codes(width,64);matrix=codes.flatten(0,1).cuda();scale=float(matrix[0,0].abs())
  perm,inv,signs=[z.cuda() for z in compact_codes(width,64)]
  x=torch.randn(256,width,device='cuda');u=torch.randn(256,512,device='cuda')
  cases.extend([(width,'dense_mux',lambda x=x,m=matrix:F.linear(x,m)),
                (width,'compact_mux',lambda x=x,p=perm,s=signs,v=scale:mux(x,p,s,v)),
                (width,'dense_transpose',lambda u=u,m=matrix:F.linear(u,m.T)),
                (width,'compact_transpose',lambda u=u,i=inv,s=signs,v=scale:transpose(u,i,s,v))])
 timings=[];rng=random.Random(532)
 for repeat in range(3):
  order=cases.copy();rng.shuffle(order)
  for width,name,fn in order:
   ms=triton.testing.do_bench_cudagraph(fn,rep=100)
   timings.append(dict(width=width,operator=name,repeat=repeat,ms=ms))
 summary=[dict(width=w,operator=name,median_ms=statistics.median(r['ms'] for r in timings if r['width']==w and r['operator']==name)) for w,name,_ in cases]
 a.out.with_name('isolated_projection_speed.json').write_text(json.dumps(dict(protocol='GPU0; batch256; FP32 fixed operator only, three randomized rounds per shape, 100ms captured-kernel timing per case. Transpose is the exact backward linear operator. Not full training speed.',rows=timings,summary=summary),indent=2)+'\n')

if __name__=='__main__':main()
