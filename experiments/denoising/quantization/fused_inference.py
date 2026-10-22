"""Inference-only constant folding and fused KS2 geometry; rebuild after weight updates."""
import torch
from torch import nn
from torch.nn import functional as F
class FrozenKS2(nn.Module):
 def __init__(self,trained):
  super().__init__();self.s=trained.s;self.r=trained.r
  def buf(k,v):self.register_buffer(k,v.detach().clone().contiguous())
  buf('down',trained.down.weight);buf('up',trained.up.weight)
  a0,a1,a2,a3,a4,a5=trained.geo_t.detach().unbind(1)
  for k,v in dict(a0=a0,a1=a2,a2=-3*a0+3*a1-2*a2-a3,a3=2*a0-2*a1+a2+a3,a4=a4,a5=a5).items():buf(k,v)
  q0,q1,q2,q3=trained.geo_x.detach().unbind(2)
  for k,v in dict(q0=q0-q2,q1=q1-3*q3,q2=2*q2,q3=4*q3,dd=F.softplus(trained.den_delta),dw=F.softplus(trained.den_weight),scale=F.softplus(trained.raw_scale)+1e-4,shift=.5*torch.tanh(trained.raw_shift),escale=F.softplus(trained.edge_raw_scale)+1e-4,eshift=.5*torch.tanh(trained.edge_raw_shift)).items():buf(k,v)
 def geo(self,c):
  total=c.sum(-1,keepdim=True);t=total/self.r
  near=total.abs()<1e-6;w=torch.where(near,torch.full_like(c,1/self.r),c/torch.where(near,torch.ones_like(total),total))
  z=torch.tanh(self.scale*(t+self.shift))
  p=((self.a3*t+self.a2)*t+self.a1)*t+self.a0+(2*z*z-1)*self.a4+(4*z*z*z-3*z)*self.a5
  e=torch.tanh(self.escale*(c+self.eshift))[...,None]
  ex=((self.q3*e+self.q2)*e+self.q1)*e+self.q0
  den=1+self.dd*(c-t).abs()[...,None]+self.dw*w.square()[...,None]
  return c+p[...,:self.r]+(p[...,self.r:].reshape(-1,self.s,self.r,self.r)*ex/den).sum(-2)
 def forward(self,x):
  c=F.linear(x,self.down).reshape(-1,self.s,self.r)
  return F.linear(self.geo(c).reshape(-1,self.s*self.r),self.up)

class NativeKS2(FrozenKS2):
 """CPU float32 rank-8 inference only. Uses AVX2 and glibc libmvec; no backward."""
 def __init__(self,trained,threads=1):
  super().__init__(trained)
  import ctypes
  from pathlib import Path
  if self.r!=8 or trained.h!=1 or trained.combine!='sum':raise ValueError('Native kernel requires rank8, one head, sum combination')
  self.threads=threads;self._lib=ctypes.CDLL(str(Path(__file__).with_name('ks2_fused.so')))
  self._fn=self._lib.ks2_geo;self._fn.argtypes=[ctypes.c_void_p]*7+[ctypes.c_int]*3;self._fn.restype=None
  self.register_buffer('packed_a',torch.stack([getattr(self,f'a{i}') for i in range(6)],1).contiguous())
  self.register_buffer('packed_q',torch.stack([getattr(self,f'q{i}') for i in range(4)],1).contiguous())
  self.register_buffer('warp',torch.stack([self.scale.flatten(),self.shift.flatten(),self.escale.flatten(),self.eshift.flatten()],1).contiguous())
 def geo(self,c):
  if torch.is_grad_enabled():raise RuntimeError('NativeKS2 is inference only; use torch.no_grad()')
  if c.device.type!='cpu' or c.dtype!=torch.float32 or not c.is_contiguous():raise ValueError('Requires contiguous CPU float32')
  y=torch.empty_like(c)
  self._fn(c.data_ptr(),self.packed_a.data_ptr(),self.packed_q.data_ptr(),self.dd.data_ptr(),self.dw.data_ptr(),self.warp.data_ptr(),y.data_ptr(),c.shape[0],self.s,self.threads)
  return y
