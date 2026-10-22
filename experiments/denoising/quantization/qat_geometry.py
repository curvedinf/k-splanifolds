"""True quantized integer forward, floating-point STE backward, FP32 masters."""
import torch
from torch.nn import functional as F
from qat_base import IntegerGeometryBase

def substitute(exact,surrogate):
 # Exact forward values, approximate backward through the surrogate.
 return exact.detach()+(surrogate-surrogate.detach())
class QATGeometry(IntegerGeometryBase):
 def __init__(self,mode):
  super().__init__(mode)
  for name in ['basis','atten','ex']:
   self.register_buffer('range_'+name,torch.zeros(self.s))
  self.quant_steps=0
 def observe(self,name,x):
  buf=getattr(self,'range_'+name)
  if self.training:
   with torch.no_grad():
    v=x.detach().float().transpose(0,1).reshape(self.s,-1).abs().amax(1)*1.2
    if not bool(buf.any()):buf.copy_(v)
    else:buf.mul_(.95).add_(v,alpha=.05)
  self.ranges[name]=buf.detach().clone().clamp_min(1e-8)
 def forward(self,x):
  self.prepare_weights() # Repack current master weights after optimizer updates.
  return super().forward(x)
 def fake_activation(self,name,x):
  q,scale=self.quant_activation(name,x)
  shape=(1,self.s)+(1,)*(x.ndim-2);limit=(scale*127).reshape(shape)
  clipped=torch.maximum(torch.minimum(x.float(),limit),-limit)
  return substitute(q.float()*scale.reshape(shape),clipped)
 def fake_weights(self,w):
  flat=w.float().reshape(self.s,-1);width=flat.shape[1]
  blocks=F.pad(flat,(0,(-width)%128)).reshape(self.s,-1,128)
  with torch.no_grad():
   scale=blocks.detach().abs().amax(-1,keepdim=True).clamp_min(1e-8)/127
   dq=(blocks.detach()/scale).round().clamp(-127,127)*scale
  return substitute(dq,blocks).reshape(self.s,-1)[:,:width].reshape(w.shape)
 def contract_t(self,basis):
  exact=super().contract_t(basis)
  if not torch.is_grad_enabled():return exact
  sur=torch.bmm(self.fake_activation('basis',basis).transpose(0,1),self.fake_weights(self.geo_t)).transpose(0,1)
  return substitute(exact,sur.half())
 def contract_x(self,eb):
  exact=super().contract_x(eb)
  if not torch.is_grad_enabled():return exact
  q=(eb.detach().float()*127).round().clamp(-127,127)/127
  fake=substitute(q,eb.float().clamp(-1,1))
  sur=torch.bmm(fake.permute(1,2,0,3).reshape(self.s*8,-1,4),self.fake_weights(self.geo_x).reshape(self.s*8,4,8)).reshape(self.s,8,-1,8).permute(2,0,1,3)
  return substitute(exact,sur.half())
 def edge_reduce(self,atten,ex,qa,sa,qx,sx):
  exact=super().edge_reduce(atten,ex,qa,sa,qx,sx)
  if not torch.is_grad_enabled():return exact
  sur=(self.fake_activation('atten',atten)*self.fake_activation('ex',ex)).sum(2)
  return substitute(exact,sur.half())
 def powers(self,x,bound):
  exact2,exact3=super().powers(x,bound)
  if not torch.is_grad_enabled():return exact2,exact3
  q=(x.detach().float()*(127/bound)).round().clamp(-127,127)*(bound/127)
  fake=substitute(q,x.float().clamp(-bound,bound))
  two=substitute(exact2,fake.square().half())
  three=substitute(exact3,(two.float()*fake).half())
  return two,three

def make_model(kind,seed):
 torch.manual_seed(seed);return QATGeometry('combined' if kind=='qat_combined' else 'powers_all')
