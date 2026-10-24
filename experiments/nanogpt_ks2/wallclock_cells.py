"""Capacity-preserving chart/basis experiments for the nanoGPT research harness."""
from types import MethodType
import math

import torch
from torch import nn

from experiments.common.ks3_int8 import ks3_int8


class MixedKS3(nn.Module):
    """Several chart ranks sharing one pair of trainable projection matrices.

    Banks are (chart count, rank) pairs. Every coordinate and chart parameter is
    used; the same full matrix optimizer applies across all banks.
    """
    def __init__(self,width,banks):
        super().__init__()
        self.banks=tuple((int(count),int(rank)) for count,rank in banks)
        if any(min(count,rank)<1 for count,rank in self.banks):
            raise ValueError('Positive chart counts and ranks required')
        self.features=sum(count*rank for count,rank in self.banks)
        self.num_subspaces=sum(count for count,rank in self.banks)
        self.down=nn.Linear(width,self.features,bias=False)
        self.up=nn.Linear(self.features,width,bias=False)
        self.coefficients=nn.Parameter(torch.empty(self.features,6))
        self.raw_scale=nn.Parameter(torch.full((self.num_subspaces,),.75))
        self.raw_shift=nn.Parameter(torch.zeros(self.num_subspaces))
        nn.init.normal_(self.coefficients,std=.05)
        bound=math.sqrt(3/width)*.5
        nn.init.uniform_(self.down.weight,-bound,bound)
        nn.init.zeros_(self.up.weight)

    def forward(self,inputs):
        output=None;feature=chart=0
        for count,rank in self.banks:
            stop=feature+count*rank;end=chart+count
            coefficients=self.coefficients[feature:stop]
            if rank!=1:coefficients=coefficients.view(count,rank,6)
            part=ks3_int8(inputs,self.down.weight[feature:stop],self.up.weight[:,feature:stop],
                          coefficients,self.raw_scale[chart:end],self.raw_shift[chart:end],
                          subspace_rank=rank,nonlinear='fp16')
            output=part if output is None else output+part
            feature=stop;chart=end
        return output


def to_hermite(q,basis):
    q0,q1,q2,q3,w4,w5=q.unbind(-1)
    scale=2. if basis in ('power2','chebyshev2') else 1.
    if basis.startswith('power'):
        d,c,b,a=q0,q1/scale,q2/scale**2,q3/scale**3
    elif basis=='chebyshev2':
        d=q0-q2;c=(q1-3*q3)/scale;b=2*q2/scale**2;a=4*q3/scale**3
    else:
        raise ValueError(basis)
    return torch.stack((d,a+b+c+d,c,3*a+2*b+c,w4,w5),dim=-1)


def _basis_forward(self,inputs):
    return ks3_int8(inputs,self.down.weight,self.up.weight,
                    to_hermite(self.coefficients,self._research_basis),
                    self.raw_scale,self.raw_shift,subspace_rank=self.subspace_rank,
                    nonlinear=self._ks3_nonlinear,tanh_lut=self._ks3_tanh_lut)


@torch.no_grad()
def change_basis(cell,basis):
    p0,p1,p2,p3,w4,w5=cell.coefficients.unbind(-1)
    a=2*p0-2*p1+p2+p3;b=-3*p0+3*p1-2*p2-p3;c=p2;d=p0
    scale=2. if basis in ('power2','chebyshev2') else 1.
    if basis.startswith('power'):
        values=(d,c*scale,b*scale**2,a*scale**3,w4,w5)
    elif basis=='chebyshev2':
        values=(d+b*scale**2/2,c*scale+3*a*scale**3/4,b*scale**2/2,a*scale**3/4,w4,w5)
    else:
        raise ValueError(basis)
    cell.coefficients.copy_(torch.stack(values,dim=-1))
    cell._research_basis=basis
    cell.forward=MethodType(_basis_forward,cell)
    return cell
