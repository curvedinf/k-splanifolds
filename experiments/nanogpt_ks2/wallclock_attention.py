"""Exact causal/document/window attention with linear work in sequence length.

The reference port materializes a dense T×T attention mask. These FP32 kernels
visit only the allowed local window. Backward owns each output gradient row,
so it uses no atomic additions and retains deterministic execution.
"""
from types import MethodType

import torch
import triton as tr
import triton.language as tl
from torch.nn import functional as F


@tr.jit
def _forward(Q,K,V,DOC,O,LSE,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,E:tl.constexpr,
             QS0:tl.constexpr,QS1:tl.constexpr,QS2:tl.constexpr,QS3:tl.constexpr,
             KS0:tl.constexpr,KS1:tl.constexpr,KS2:tl.constexpr,KS3:tl.constexpr,
             VS0:tl.constexpr,VS1:tl.constexpr,VS2:tl.constexpr,VS3:tl.constexpr,
             W:tl.constexpr,SCALE:tl.constexpr,BW:tl.constexpr,BD:tl.constexpr,BE:tl.constexpr):
    i=tl.program_id(0);bh=tl.program_id(1);b=bh//H;h=bh%H
    j=i-tl.arange(0,BW);d=tl.arange(0,BD);e=tl.arange(0,BE)
    q=tl.load(Q+b*QS0+i*QS1+h*QS2+d*QS3,d<D,other=0.).to(tl.float32)
    k=tl.load(K+b*KS0+j[:,None]*KS1+h*KS2+d[None,:]*KS3,
              (j[:,None]>=0)&(d[None,:]<D),other=0.).to(tl.float32)
    doc=tl.load(DOC+i);docs=tl.load(DOC+j,j>=0,other=-1)
    valid=(j>=0)&(i-j<=W)&(doc==docs)
    score=tl.sum(q[None,:]*k,1)*SCALE
    score=tl.where(valid,score,float('-inf'))
    maximum=tl.max(score,0);p=tl.exp(score-maximum);denom=tl.sum(p,0)
    p=p/denom
    v=tl.load(V+b*VS0+j[:,None]*VS1+h*VS2+e[None,:]*VS3,
              (j[:,None]>=0)&(e[None,:]<E),other=0.).to(tl.float32)
    out=tl.sum(p[:,None]*v,0)
    tl.store(O+((b*T+i)*H+h)*E+e,out,e<E)
    tl.store(LSE+(b*T+i)*H+h,maximum+tl.log(denom))


@tr.jit
def _backward_q(Q,K,V,DOC,O,LSE,DO,DQ,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,E:tl.constexpr,
                QS0:tl.constexpr,QS1:tl.constexpr,QS2:tl.constexpr,QS3:tl.constexpr,
                KS0:tl.constexpr,KS1:tl.constexpr,KS2:tl.constexpr,KS3:tl.constexpr,
                VS0:tl.constexpr,VS1:tl.constexpr,VS2:tl.constexpr,VS3:tl.constexpr,
                W:tl.constexpr,SCALE:tl.constexpr,BW:tl.constexpr,BD:tl.constexpr,BE:tl.constexpr):
    i=tl.program_id(0);bh=tl.program_id(1);b=bh//H;h=bh%H
    j=i-tl.arange(0,BW);d=tl.arange(0,BD);e=tl.arange(0,BE)
    q=tl.load(Q+b*QS0+i*QS1+h*QS2+d*QS3,d<D,other=0.).to(tl.float32)
    k=tl.load(K+b*KS0+j[:,None]*KS1+h*KS2+d[None,:]*KS3,
              (j[:,None]>=0)&(d[None,:]<D),other=0.).to(tl.float32)
    v=tl.load(V+b*VS0+j[:,None]*VS1+h*VS2+e[None,:]*VS3,
              (j[:,None]>=0)&(e[None,:]<E),other=0.).to(tl.float32)
    doc=tl.load(DOC+i);docs=tl.load(DOC+j,j>=0,other=-1)
    valid=(j>=0)&(i-j<=W)&(doc==docs)
    score=tl.sum(q[None,:]*k,1)*SCALE
    lse=tl.load(LSE+(b*T+i)*H+h)
    p=tl.where(valid,tl.exp(score-lse),0.)
    dy=tl.load(DO+((b*T+i)*H+h)*E+e,e<E,other=0.).to(tl.float32)
    out=tl.load(O+((b*T+i)*H+h)*E+e,e<E,other=0.).to(tl.float32)
    delta=tl.sum(dy*out,0)
    ds=p*(tl.sum(v*dy[None,:],1)-delta)*SCALE
    dq=tl.sum(ds[:,None]*k,0)
    tl.store(DQ+((b*T+i)*H+h)*D+d,dq,d<D)


@tr.jit
def _backward_kv(Q,K,V,DOC,O,LSE,DO,DK,DV,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,E:tl.constexpr,
                 QS0:tl.constexpr,QS1:tl.constexpr,QS2:tl.constexpr,QS3:tl.constexpr,
                 KS0:tl.constexpr,KS1:tl.constexpr,KS2:tl.constexpr,KS3:tl.constexpr,
                 VS0:tl.constexpr,VS1:tl.constexpr,VS2:tl.constexpr,VS3:tl.constexpr,
                 W:tl.constexpr,SCALE:tl.constexpr,BW:tl.constexpr,BD:tl.constexpr,BE:tl.constexpr):
    j=tl.program_id(0);bh=tl.program_id(1);b=bh//H;h=bh%H
    i=j+tl.arange(0,BW);d=tl.arange(0,BD);e=tl.arange(0,BE)
    q=tl.load(Q+b*QS0+i[:,None]*QS1+h*QS2+d[None,:]*QS3,
              (i[:,None]<T)&(d[None,:]<D),other=0.).to(tl.float32)
    k=tl.load(K+b*KS0+j*KS1+h*KS2+d*KS3,d<D,other=0.).to(tl.float32)
    v=tl.load(V+b*VS0+j*VS1+h*VS2+e*VS3,e<E,other=0.).to(tl.float32)
    doc=tl.load(DOC+j);docs=tl.load(DOC+i,i<T,other=-1)
    valid=(i<T)&(i-j<=W)&(doc==docs)
    score=tl.sum(q*k[None,:],1)*SCALE
    lse=tl.load(LSE+(b*T+i)*H+h,i<T,other=0.)
    p=tl.where(valid,tl.exp(score-lse),0.)
    offsets=((b*T+i[:,None])*H+h)*E+e[None,:]
    dy=tl.load(DO+offsets,(i[:,None]<T)&(e[None,:]<E),other=0.).to(tl.float32)
    out=tl.load(O+offsets,(i[:,None]<T)&(e[None,:]<E),other=0.).to(tl.float32)
    delta=tl.sum(dy*out,1)
    ds=p*(tl.sum(dy*v[None,:],1)-delta)*SCALE
    dk=tl.sum(ds[:,None]*q,0);dv=tl.sum(p[:,None]*dy,0)
    tl.store(DK+((b*T+j)*H+h)*D+d,dk,d<D)
    tl.store(DV+((b*T+j)*H+h)*E+e,dv,e<E)


class _LocalAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,documents,window,scale):
        b,t,h,d=q.shape;e=v.shape[-1]
        out=torch.empty((b,t,h,e),device=q.device,dtype=v.dtype)
        lse=torch.empty((b,t,h),device=q.device,dtype=torch.float32)
        args=(t,h,d,e,*q.stride(),*k.stride(),*v.stride(),window,scale,
              tr.next_power_of_2(window+1),tr.next_power_of_2(d),tr.next_power_of_2(e))
        _forward[(t,b*h)](q,k,v,documents,out,lse,*args,num_warps=4,enable_fp_fusion=False)
        ctx.save_for_backward(q,k,v,documents,out,lse);ctx.args=args
        return out

    @staticmethod
    def backward(ctx,grad):
        q,k,v,documents,out,lse=ctx.saved_tensors
        b,t,h,d=q.shape
        grad=grad.contiguous()
        dq=torch.empty(q.shape,device=q.device,dtype=q.dtype)
        dk=torch.empty(k.shape,device=k.device,dtype=k.dtype)
        dv=torch.empty(v.shape,device=v.device,dtype=v.dtype)
        _backward_q[(t,b*h)](q,k,v,documents,out,lse,grad,dq,*ctx.args,num_warps=4,enable_fp_fusion=False)
        _backward_kv[(t,b*h)](q,k,v,documents,out,lse,grad,dk,dv,*ctx.args,num_warps=4,enable_fp_fusion=False)
        return dq,dk,dv,None,None,None


def local_attention(q,k,v,documents,window,scale):
    return _LocalAttention.apply(q,k,v,documents,window,scale)


def _attention_forward(self,x,a,qk_w,v_w,o_w,unused=None):
    from experiments.nanogpt_ks2 import model as reference
    b,t,_=x.shape;h=6
    qk=F.linear(x,qk_w*a.sa_lambdas[0]).view(b,t,2*h,self.qk_dim)
    q,k=reference.norm(qk[:,:,:h]),reference.norm(qk[:,:,h:])
    q,k=a.yarn.rotate(q),a.yarn.rotate(k)
    if a.key_offset:
        n=reference.NARROW_HEAD_DIM
        k=torch.cat((k[...,:n],torch.cat((k[:,:1,:,n:],k[:,:-1,:,n:]),dim=1)),dim=-1)
    v=F.linear(x,v_w*a.sa_lambdas[0]).view(b,t,h,self.v_dim)
    if a.aux_v is not None:v=v+a.aux_v.view(b,t,h,self.head_dim)[...,:self.v_dim]
    documents=a.seqlens
    if self.paired:
        q=q.reshape(b,t*2,h//2,self.qk_dim);k=k.reshape(b,t*2,h//2,self.qk_dim)
        v=v.reshape(b,t*2,h//2,self.v_dim);documents=documents.repeat_interleave(2)
    y=local_attention(q,k,v,documents,a.bm_size,a.yarn.attn_scale).reshape(b,t,h,self.v_dim)
    if a.xsa_alpha is not None and not self.paired:
        y=y-a.xsa_alpha.tanh().view(b,t,h,1)*(y*v).sum(-1,keepdim=True)/v.square().sum(-1,keepdim=True).clamp_min(1e-8)*v
    if a.attn_gate_w is not None:y=y*a.attn_gate_w.view(b,t,h,1)
    gain=a.sa_lambdas[1]*(1 if a.o_gain is None else a.o_gain)
    return F.linear(y.reshape(b,t,h*self.v_dim),o_w*gain)


def install_window_attention(model):
    for attention in model.attn.values():
        attention.forward=MethodType(_attention_forward,attention)
    return model
