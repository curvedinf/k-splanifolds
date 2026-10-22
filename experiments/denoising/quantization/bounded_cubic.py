"""FP16 inner geometry ablation; FP32 projections/master parameters/optimizer.
Stable version replaces signed-sum damping with bounded energy/distance predicates.
"""
import math
import torch
from torch.nn import functional as F
from parallel_models import ParallelKS2
class BoundedCubicFP16(ParallelKS2):
 def __init__(self,stable=True):
  super().__init__(32,1,'sum',r=8);self.stable=stable;self.capture=False;self.diagnostics={}
  self.strength_offset=math.log(math.log1p(math.exp(-4))/(8-math.log1p(math.exp(-4))))+4
 def geo(self,c):
  c=c.half();b,u,r=c.shape
  # Shared FP32 chart reduction; round the longitudinal coordinate to FP16.
  cf=c.float();total=cf.sum(-1,keepdim=True)
  # Fixed smooth bound: unit slope at zero, |t| <= 4, |t^3| <= 64.
  t=(4*torch.tanh(total/(r*4))).half()
  z=torch.tanh((F.softplus(self.raw_scale)+1e-4).half()*(t+(.5*torch.tanh(self.raw_shift)).half()))
  t2=t*t;t3=t2*t
  basis=torch.cat((2*t3-3*t2+1,-2*t3+3*t2,t3-2*t2+t,t3-t2,2*z*z-1,4*z*z*z-3*z),-1)
  packed=torch.bmm(basis.transpose(0,1),self.geo_t.half()).transpose(0,1)
  spine=packed[...,:r];et=packed[...,r:].reshape(b,u,r,r)
  e=torch.tanh((F.softplus(self.edge_raw_scale)+1e-4).half()*(c+(.5*torch.tanh(self.edge_raw_shift)).half()))
  eb=torch.stack((torch.ones_like(e),e,2*e*e-1,4*e*e*e-3*e),-1)
  ex=torch.bmm(eb.permute(1,2,0,3).reshape(u*r,b,4),self.geo_x.half().reshape(u*r,4,r)).reshape(u,r,b,r).permute(2,0,1,3)
  if self.stable:
   mag=cf.abs().amax(-1,keepdim=True);cn=cf/torch.where(mag>0,mag,torch.ones_like(mag))
   energy=cn.square();es=energy.sum(-1,keepdim=True)
   importance=energy/torch.where(es>0,es,torch.ones_like(es))
   deviation=cn-cn.mean(-1,keepdim=True);ds=deviation.square();div=ds+es/r
   distance=ds/torch.where(div>0,div,torch.ones_like(div))
   aa=(8*torch.sigmoid(self.den_delta+self.strength_offset)).half()
   bb=(8*torch.sigmoid(self.den_weight+self.strength_offset)).half()
   den=1+aa*distance.half()[...,None]+bb*importance.half()[...,None]
   # Apply attenuation before multiplying edge factors.
   edges=(et/den)*ex
  else:
   near=total.abs()<1e-6
   w=(cf/torch.where(near,torch.ones_like(total),total)).half()
   w=torch.where(near,torch.full_like(w,1/r),w)
   den=1+F.softplus(self.den_delta).half()*(c-t).abs()[...,None]+F.softplus(self.den_weight).half()*w.square()[...,None]
   edges=et*ex/den
  acc=torch.zeros_like(c)
  for i in range(r):acc=acc+edges[:,:,i,:]
  geo=spine+acc
  if self.capture:
   self.diagnostics={k:dict(nonfinite=int((~torch.isfinite(v)).sum()),elements=v.numel(),max_abs=float(v.detach().abs().max())) for k,v in {'coordinates':c,'basis_coordinate':t,'basis':basis,'polynomial':packed,'coordinate_edges':ex,'denominator':den,'edges':edges,'geometry':geo}.items()}
   self.diagnostics['denominator_min']=float(den.detach().min())
  # Round the latent residual addition to half as well.
  return ((c+geo).float()-cf)
 def forward(self,x):
  c=self.down(x).reshape(-1,self.s,self.r)
  # geo returns a correction relative to rounded c; direct formulation avoids another residual.
  return self.up((c.half().float()+self.geo(c)).reshape(-1,self.s*self.r))
def make_model(kind,seed):
 torch.manual_seed(seed);return BoundedCubicFP16(stable=True)
