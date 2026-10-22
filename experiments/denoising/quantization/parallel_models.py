"""Parallel banks of full multivariate rank-8 KS2 units.
Preserves output-specific multiplicative edges, rational damping and residuals.
Packed spine/edge-t bmm and batched edge-x bmm; no sequential refinement.
"""
import torch
from torch import nn
from torch.nn import functional as F

class ParallelKS2(nn.Module):
    def __init__(self,subspaces,heads,combine,d=169,r=8):
        super().__init__();self.s=subspaces;self.h=heads;self.r=r;self.combine=combine
        u=subspaces*heads
        self.down=nn.Linear(d,subspaces*r,bias=False)
        self.up=nn.Linear(subspaces*r*(heads if combine=='concat' else 1),d,bias=False)
        # Packing concatenates the original independent spine and edge-t coefficients.
        self.geo_t=nn.Parameter(torch.randn(u,6,r+r*r)*.05)
        self.geo_x=nn.Parameter(torch.randn(u,r,4,r)*.05)
        self.den_delta=nn.Parameter(torch.full((u,r,r),-4.))
        self.den_weight=nn.Parameter(torch.full((u,r,r),-4.))
        self.raw_scale=nn.Parameter(torch.full((1,u,1),.75))
        self.raw_shift=nn.Parameter(torch.zeros(1,u,1))
        self.edge_raw_scale=nn.Parameter(torch.full((1,u,1),.75))
        self.edge_raw_shift=nn.Parameter(torch.zeros(1,u,1))
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
        edge_basis=torch.stack((torch.ones_like(e),e,2*e*e-1,4*e*e*e-3*e),-1)
        # Each (unit,input-coordinate) owns a 4-by-r coefficient matrix.
        eb=edge_basis.permute(1,2,0,3).reshape(u*r,b,4)
        edge_x=torch.bmm(eb,self.geo_x.reshape(u*r,4,r)).reshape(u,r,b,r).permute(2,0,1,3)
        damping=1+F.softplus(self.den_delta)[None]*delta.abs()[...,None]+F.softplus(self.den_weight)[None]*weights.square()[...,None]
        return spine+(edge_t*edge_x/damping).sum(-2)
    def forward(self,x):
        b=x.shape[0];c=self.down(x).reshape(b,self.s,self.r)
        # Different heads see identical original coordinates, never each other's output.
        unit_coords=c[:,:,None,:].expand(b,self.s,self.h,self.r).reshape(b,self.s*self.h,self.r)
        g=self.geo(unit_coords).reshape(b,self.s,self.h,self.r)
        if self.combine=='concat':y=c[:,:,None,:]+g
        else:y=c+g.sum(2)
        return self.up(y.reshape(b,-1))

CONFIGS={'KS2Single':dict(subspaces=123,heads=1,combine='sum'),
         'KS2ParallelSum':dict(subspaces=100,heads=2,combine='sum'),
         'KS2ParallelConcat':dict(subspaces=76,heads=2,combine='concat')}
CLASSES=CONFIGS
def make_model(kind,seed):
    torch.manual_seed(seed);return ParallelKS2(**CONFIGS[kind])
