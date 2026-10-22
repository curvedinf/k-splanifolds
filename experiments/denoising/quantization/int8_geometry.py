"""FP16 inner geometry ablation; FP32 projections/master parameters/optimizer.
Stable version replaces signed-sum damping with bounded energy/distance predicates.
"""
import math
import torch
from torch.nn import functional as F
from parallel_models import ParallelKS2
class IntegerGeometry(ParallelKS2):
 def __init__(self,mode="reference"):
  stable=True;self.mode=mode;self.calibrating=False;self.ranges={};self.clips={};self.count_clips=False
  super().__init__(32,1,'sum',r=8);self.stable=stable;self.capture=False;self.diagnostics={}
  self.strength_offset=math.log(math.log1p(math.exp(-4))/(8-math.log1p(math.exp(-4))))+4
 def geo(self,c):
  c=c.half();b,u,r=c.shape
  # Shared FP32 chart reduction; round the longitudinal coordinate to FP16.
  cf=c.float();total=cf.sum(-1,keepdim=True)
  # Fixed smooth bound: unit slope at zero, |t| <= 4, |t^3| <= 64.
  t=(4*torch.tanh(total/(r*4))).half()
  z=torch.tanh((F.softplus(self.raw_scale)+1e-4).half()*(t+(.5*torch.tanh(self.raw_shift)).half()))
  t2,t3=self.powers(t,4) if self.mode=="powers_all" else (t*t,t*t*t)
  z2,z3=self.powers(z,1) if self.mode=="powers_all" else (z*z,z*z*z)
  basis=torch.cat((2*t3-3*t2+1,-2*t3+3*t2,t3-2*t2+t,t3-t2,2*z2-1,4*z3-3*z),-1)
  self.observe("basis",basis)
  packed=self.contract_t(basis) if self.mode in ["contracts","combined","powers_all"] and not self.calibrating else torch.bmm(basis.transpose(0,1),self.geo_t.half()).transpose(0,1)
  spine=packed[...,:r];et=packed[...,r:].reshape(b,u,r,r)
  e=torch.tanh((F.softplus(self.edge_raw_scale)+1e-4).half()*(c+(.5*torch.tanh(self.edge_raw_shift)).half()))
  e2,e3=self.powers(e,1) if self.mode=="powers_all" else (e*e,e*e*e)
  eb=torch.stack((torch.ones_like(e),e,2*e2-1,4*e3-3*e),-1)
  ex=torch.bmm(eb.permute(1,2,0,3).reshape(u*r,b,4),self.geo_x.half().reshape(u*r,4,r)).reshape(u,r,b,r).permute(2,0,1,3)
  if self.mode in ["contracts","combined","powers_all"] and not self.calibrating:ex=self.contract_x(eb)
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
   atten=et/den
   self.observe("atten",atten);self.observe("ex",ex)
   edges=atten*ex
  else:
   near=total.abs()<1e-6
   w=(cf/torch.where(near,torch.ones_like(total),total)).half()
   w=torch.where(near,torch.full_like(w,1/r),w)
   den=1+F.softplus(self.den_delta).half()*(c-t).abs()[...,None]+F.softplus(self.den_weight).half()*w.square()[...,None]
   edges=et*ex/den
  acc=torch.zeros_like(c)
  for i in range(r):acc=acc+edges[:,:,i,:]
  if self.mode in ["edges","combined","powers_all"] and not self.calibrating:
   qa,sa=self.quant_activation("atten",atten);qx,sx=self.quant_activation("ex",ex)
   acc=((qa.int()*qx.int()).sum(2,dtype=torch.int32).float()*(sa*sx).reshape(1,self.s,1)).half()
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
 def observe(self,name,x):
  if self.calibrating:
   v=x.float().transpose(0,1).reshape(self.s,-1).abs().amax(1)
   self.ranges[name]=torch.maximum(self.ranges.get(name,torch.zeros_like(v)),v)
 def quant_activation(self,name,x):
  scale=self.ranges[name].clamp_min(1e-8)/127
  shape=(1,self.s)+(1,)*(x.ndim-2);v=x.float()/scale.reshape(shape)
  if self.count_clips:
   d=self.clips.setdefault(name,[0,0]);d[0]+=int((v.abs()>127).sum());d[1]+=v.numel()
  return v.round().clamp(-127,127).to(torch.int8),scale
 def prepare_weights(self):
  def pack(w):
   flat=w.detach().float().reshape(self.s,-1);width=flat.shape[1]
   blocks=F.pad(flat,(0,(-width)%128)).reshape(self.s,-1,128)
   scale=blocks.abs().amax(-1).clamp_min(1e-8)/127
   q=(blocks/scale[...,None]).round().clamp(-127,127).to(torch.int8).reshape(self.s,-1)[:,:width]
   return q,scale
  qt,self.ts=pack(self.geo_t);self.qt=[]
  for g in range(self.ts.shape[1]):
   mask=(torch.arange(qt.shape[1])//128==g)
   self.qt.append((qt*mask).reshape(self.s,6,72).int())
  qx,self.xs=pack(self.geo_x);self.qx=qx.reshape(self.s*8,4,8).int()
 def contract_t(self,basis):
  qa,sa=self.quant_activation('basis',basis);qa=qa.transpose(0,1).contiguous().int()
  out=torch.zeros(basis.shape[0],self.s,72)
  for g,w in enumerate(self.qt):
   dot=torch.bmm(qa,w).transpose(0,1)
   out+=dot.float()*(sa*self.ts[:,g]).reshape(1,self.s,1)
  return out.half()
 def contract_x(self,eb):
  # Bounded Chebyshev basis uses known [-1,1] range; FP16 rounding can slightly exceed it.
  qa=(eb.float()*127).round().clamp(-127,127).to(torch.int8)
  qa=qa.permute(1,2,0,3).reshape(self.s*8,-1,4).int()
  dot=torch.bmm(qa,self.qx).reshape(self.s,8,-1,8).permute(2,0,1,3)
  scale=self.xs.repeat_interleave(4,dim=1).reshape(1,self.s,8,1)/127
  return (dot.float()*scale).half()
 def powers(self,x,bound):
  # INT8 operands, INT32 products, round/requantize before the next multiply.
  q=(x.float()*(127/bound)).round().clamp(-127,127).to(torch.int8)
  q2=((q.int()*q.int()).float()/127).round().clamp(0,127).to(torch.int8)
  q3=((q2.int()*q.int()).float()/127).round().clamp(-127,127).to(torch.int8)
  return (q2.float()*(bound*bound/127)).half(),(q3.float()*(bound**3/127)).half()
