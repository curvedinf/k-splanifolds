"""Exact Horner evaluation of complete rank-8 KS2 units, suitable for fusion."""
import torch
from torch.nn import functional as F
from parallel_models import ParallelKS2

class HornerKS2(ParallelKS2):
    def __init__(self,subspaces=123,rank=8):super().__init__(subspaces,1,'sum',r=rank)
    def geo(self,c):
        b,u,r=c.shape
        total=c.sum(-1,keepdim=True);t=total/r;delta=c-t
        near=total.abs()<1e-6
        weights=c/torch.where(near,torch.ones_like(total),total)
        weights=torch.where(near,torch.full_like(weights,1/r),weights)
        z=torch.tanh((F.softplus(self.raw_scale)+1e-4)*(t+.5*torch.tanh(self.raw_shift)))
        a0,a1,a2,a3,a4,a5=self.geo_t.unbind(1)
        # Exactly the original four Hermite and two warped Chebyshev terms.
        packed=(((2*a0-2*a1+a2+a3)*t+(-3*a0+3*a1-2*a2-a3))*t+a2)*t+a0+(2*z*z-1)*a4+(4*z*z*z-3*z)*a5
        spine=packed[...,:r];edge_t=packed[...,r:].reshape(b,u,r,r)
        e=torch.tanh((F.softplus(self.edge_raw_scale)+1e-4)*(c+.5*torch.tanh(self.edge_raw_shift)))[...,None]
        q0,q1,q2,q3=self.geo_x.unbind(2)
        edge_x=((4*q3*e+2*q2)*e+(q1-3*q3))*e+(q0-q2)
        den=1+F.softplus(self.den_delta)[None]*delta.abs()[...,None]+F.softplus(self.den_weight)[None]*weights.square()[...,None]
        return spine+(edge_t*edge_x/den).sum(-2)

CONFIGS={'Fused123':123,'Fused64':64,'Fused32':32}
def make_eager(name,seed):
    torch.manual_seed(seed);return HornerKS2(CONFIGS[name])
def compile_model(m):return torch.compile(m,fullgraph=True,dynamic=False)

def edge_reduce(a,b,delta,weights,dd,dw):
    den=1+F.softplus(dd)[None]*delta.abs()[...,None]+F.softplus(dw)[None]*weights.square()[...,None]
    return (a*b/den).sum(-2)

fused_edge_reduce=torch.compile(edge_reduce,fullgraph=True,dynamic=False)

class SelectiveKS2(ParallelKS2):
    def __init__(self,subspaces=123,rank=8):super().__init__(subspaces,1,'sum',r=rank)
    def geo(self,c):
        b,u,r=c.shape
        total=c.sum(-1,keepdim=True);t=total/r;delta=c-t
        near=total.abs()<1e-6
        weights=c/torch.where(near,torch.ones_like(total),total)
        weights=torch.where(near,torch.full_like(weights,1/r),weights)
        z=torch.tanh((F.softplus(self.raw_scale)+1e-4)*(t+.5*torch.tanh(self.raw_shift)))
        t2=t*t;t3=t2*t
        basis=torch.cat((2*t3-3*t2+1,-2*t3+3*t2,t3-2*t2+t,t3-t2,2*z*z-1,4*z*z*z-3*z),-1)
        packed=torch.bmm(basis.transpose(0,1),self.geo_t).transpose(0,1)
        spine=packed[...,:r];edge_t=packed[...,r:].reshape(b,u,r,r)
        e=torch.tanh((F.softplus(self.edge_raw_scale)+1e-4)*(c+.5*torch.tanh(self.edge_raw_shift)))
        eb=torch.stack((torch.ones_like(e),e,2*e*e-1,4*e*e*e-3*e),-1)
        edge_x=torch.bmm(eb.permute(1,2,0,3).reshape(u*r,b,4),self.geo_x.reshape(u*r,4,r)).reshape(u,r,b,r).permute(2,0,1,3)
        return spine+fused_edge_reduce(edge_t,edge_x,delta,weights,self.den_delta,self.den_weight)

def make_selective(name,seed):
    torch.manual_seed(seed);return SelectiveKS2(CONFIGS[name])
