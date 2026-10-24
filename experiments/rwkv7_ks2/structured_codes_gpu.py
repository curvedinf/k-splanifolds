"""FP32 block-Walsh mux and transpose with compact integer codes, no atomics."""
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable


@tr.jit
def _mux(X,P,S,Y,W:tl.constexpr,C:tl.constexpr,SCALE:tl.constexpr,G:tl.constexpr):
    n=tl.program_id(0);c=tl.program_id(1)
    g=tl.arange(0,G);j=tl.arange(0,8);k=g[:,None]*8+j[None,:];valid=k<W
    p=tl.load(P+c*W+k,valid,other=0).to(tl.int32)
    a=tl.load(X+n*W+p,valid,other=0)
    for h in tl.static_range(3):
        partner=tl.gather(a,tl.broadcast_to((j^(1<<h))[None,:],(G,8)),1)
        a=tl.where((j[None,:]&(1<<h))==0,a+partner,partner-a)
    sign=tl.load(S+c*W+k,valid,other=0).to(tl.float32)
    out=tl.sum(a*sign,0)*SCALE
    tl.store(Y+n*C*8+c*8+j,out)


@tr.jit
def _transpose(X,I,S,Y,W:tl.constexpr,C:tl.constexpr,SCALE:tl.constexpr,CH:tl.constexpr,B:tl.constexpr):
    n=tl.program_id(0);block=tl.program_id(1)
    c=tl.arange(0,CH);i=block*B+tl.arange(0,B);j=tl.arange(0,8)
    valid=(c[:,None]<C)&(i[None,:]<W)
    inv=tl.load(I+c[:,None]*W+i[None,:],valid,other=0).to(tl.int32)
    group=inv//8;local=inv%8
    idx=c[:,None,None]*W+group[:,:,None]*8+j[None,None,:]
    signs=tl.load(S+idx,valid[:,:,None],other=0).to(tl.float32)
    bits=local[:,:,None]&j[None,None,:]
    parity=(bits^(bits>>1)^(bits>>2))&1
    weight=tl.where(parity==0,signs,-signs)
    x=tl.load(X+n*C*8+c[:,None]*8+j[None,:],c[:,None]<C,other=0)
    y=tl.sum(tl.sum(weight*x[:,None,:],2),0)*SCALE
    tl.store(Y+n*W+i,y,i<W)


def mux(x,perm,signs,scale):
    width=perm.shape[1];charts=perm.shape[0];n=x.numel()//width
    x=x.contiguous();y=torch.empty((*x.shape[:-1],charts*8),device=x.device,dtype=x.dtype)
    _mux[(n,charts)](x,perm,signs,y,width,charts,scale,tr.next_power_of_2(width//8),num_warps=4,enable_fp_fusion=False)
    return y


def transpose(x,inverse,signs,scale):
    charts,width=inverse.shape;n=x.numel()//(charts*8)
    x=x.contiguous();y=torch.empty((*x.shape[:-1],width),device=x.device,dtype=x.dtype)
    _transpose[(n,tr.cdiv(width,16))](x,inverse,signs,y,width,charts,scale,tr.next_power_of_2(charts),16,num_warps=4,enable_fp_fusion=False)
    return y


class Mux(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,perm,inverse,signs,scale):
        assert x.dtype==torch.float32
        ctx.save_for_backward(perm,inverse,signs);ctx.scale=scale
        return mux(x,perm,signs,scale)
    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        perm,inverse,signs=ctx.saved_tensors
        return transpose(grad,inverse,signs,ctx.scale),None,None,None,None


class Demux(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,perm,inverse,signs,scale):
        assert x.dtype==torch.float32
        ctx.save_for_backward(perm,inverse,signs);ctx.scale=scale
        return transpose(x,inverse,signs,scale)
    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        perm,inverse,signs=ctx.saved_tensors
        return mux(grad,perm,signs,ctx.scale),None,None,None,None
