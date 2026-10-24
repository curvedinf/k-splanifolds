"""Compact exact signed-code representation for the existing block Hadamard mux.

The code is a permuted blockwise Walsh transform followed by signs and folding.
This is not a full-width Hadamard transform. CUDA kernels are optional.
"""
import math
import torch
from torch import nn
from spread_projection import make_codes


def walsh(x):
    r=x.shape[-1];h=1
    while h<r:
        shape=x.shape;x=x.reshape(*shape[:-1],-1,2,h)
        a,b=x.unbind(-2);x=torch.stack((a+b,a-b),-2).reshape(shape);h*=2
    return x


def compact_codes(width,charts,rank=8,seed=4242+99173):
    assert rank==8 and width<32768
    gen=torch.Generator().manual_seed(seed)
    perm=torch.stack([torch.randperm(width,generator=gen) for _ in range(charts)])
    signs=torch.stack([torch.randint(0,2,(width,),generator=gen,dtype=torch.int8)*2-1 for _ in range(charts)])
    return perm.to(torch.int16),perm.argsort(1).to(torch.int16),signs


def reference_mux(x,perm,signs,scale):
    width=perm.shape[-1];shape=x.shape[:-1];charts=perm.shape[0]
    x=x.reshape(-1,width)[:,perm.long()].reshape(-1,charts,width//8,8)
    y=walsh(x)*signs.reshape(1,charts,width//8,8)
    return (y.sum(-2)*scale).reshape(*shape,charts*8)


def reference_demux(x,inv,signs,scale):
    charts,width=inv.shape;shape=x.shape[:-1]
    y=x.reshape(-1,charts,1,8)*signs.reshape(1,charts,width//8,8)
    y=walsh(y).reshape(-1,charts,width)
    y=y.gather(2,inv.long().unsqueeze(0).expand(y.shape[0],-1,-1))
    return (y.sum(1)*scale).reshape(*shape,width)


class CodeBook(nn.Module):
    def __init__(self,width,charts):
        super().__init__()
        perm,inv,signs=compact_codes(width,charts)
        for name,value in [('perm',perm),('inverse',inv),('signs',signs)]:self.register_buffer(name,value)


class StructuredInput(nn.Module):
    def __init__(self,base,charts,book=None):
        super().__init__();width=base.weight.shape[1]
        magnitudes=base.weight.abs().unique();assert len(magnitudes)==1
        self.scale=float(magnitudes[0]);self.width=width;self.charts=charts
        self.codes=book if book is not None else CodeBook(width,charts)
    def forward(self,x):
        if not x.is_cuda:return reference_mux(x,self.codes.perm,self.codes.signs,self.scale)
        from structured_codes_gpu import Mux
        return Mux.apply(x,self.codes.perm,self.codes.inverse,self.codes.signs,self.scale)


class StructuredOutput(nn.Module):
    def __init__(self,base,charts,book=None):
        super().__init__();width=base.matrix.shape[0]
        magnitudes=base.matrix.abs().unique();assert len(magnitudes)==1
        self.scale=float(magnitudes[0]);self.width=width;self.charts=charts
        self.weight=base.weight
        self.codes=book if book is not None else CodeBook(width,charts)
    def forward(self,x):
        if x.is_cuda:
            from structured_codes_gpu import Demux
            y=Demux.apply(x,self.codes.perm,self.codes.inverse,self.codes.signs,self.scale)
        else:y=reference_demux(x,self.codes.inverse,self.codes.signs,self.scale)
        return y*self.weight


def install(model):
    books={}
    def book(width,charts):
        key=(width,charts)
        if key not in books:books[key]=CodeBook(width,charts)
        return books[key]
    for block in model.blocks:
        cell=block.ffn
        assert cell.up.matrix.shape[0]==block.output_width, "Install before final-side pruning"
        cell.down=StructuredInput(cell.down,cell.num_subspaces,book(cell.down.weight.shape[1],cell.num_subspaces))
        cell.up=StructuredOutput(cell.up,cell.num_subspaces,book(cell.up.matrix.shape[0],cell.num_subspaces))
    return model
