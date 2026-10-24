"""Full-window 1-bit QKV ROSA, with exact single-bit counterfactual credits.

Forward follows BlinkDL's RWKV-v8/251105_reverse_run.py: longest suffix,
latest endpoint in strictly previous K, read its successor V; unmatched=-1.
Backward enumerates Q/K flips exactly for the short T=64 experiment, then
maps credits to logits with sigmoid(logit)*(1-sigmoid(logit)), tau=1.
This is a discrete counterfactual surrogate, not a derivative of hard signs.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _retrieve(qword,kword,vword,T:tl.constexpr,N:tl.constexpr):
    # Latest symbol is aligned to bit zero. The trailing-zero count of XOR
    # gives the exact equal-suffix length for every causal (query,key) pair.
    i=tl.arange(0,N)
    difference=(qword>>(63-i[:,None]))^(kword>>(63-i[None,:]))
    lowest=difference&(~difference+tl.full((),1,tl.uint64))
    # lowest is an exact power of two. Rounding log2 avoids integer truncation
    # near powers; all 64 possible positions are independently tested.
    trailing=(tl.log2(lowest.to(tl.float32))+0.5).to(tl.int32)
    lengths=tl.minimum(tl.where(difference==0,64,trailing),i[None,:]+1)
    score=tl.where((i[None,:]<i[:,None])&(lengths>0),lengths*(T+1)+i[None,:],-1)
    best=tl.max(score,axis=1)
    route=tl.where(best>=0,best%(T+1)+1,-1)
    bit=((vword>>(63-tl.maximum(route,0)))&1)>0
    sign=tl.where((route>=0)&bit,1.,-1.)
    return sign,route


@tr.jit
def _forward(Q,K,V,Y,R,P,T:tl.constexpr,C:tl.constexpr,N:tl.constexpr):
    stream=tl.program_id(0);batch=stream//C;channel=stream%C
    i=tl.arange(0,N)
    offsets=batch*T*C+i*C+channel
    qword=tl.sum((tl.load(Q+offsets,i<T,other=0)>0).to(tl.uint64)<<(63-i),axis=0)
    kword=tl.sum((tl.load(K+offsets,i<T,other=0)>0).to(tl.uint64)<<(63-i),axis=0)
    vword=tl.sum((tl.load(V+offsets,i<T,other=0)>0).to(tl.uint64)<<(63-i),axis=0)
    tl.store(P+stream*3,qword);tl.store(P+stream*3+1,kword);tl.store(P+stream*3+2,vword)
    sign,route=_retrieve(qword,kword,vword,T,N)
    tl.store(Y+offsets,sign,i<T);tl.store(R+offsets,route,i<T)


@tr.jit
def _backward_qk(Q,K,P,Y,DY,DQ,DK,T:tl.constexpr,C:tl.constexpr,N:tl.constexpr):
    stream=tl.program_id(0);flip=tl.program_id(1)
    batch=stream//C;channel=stream%C
    is_key=flip>=T;position=flip%T
    qword=tl.load(P+stream*3);kword=tl.load(P+stream*3+1);vword=tl.load(P+stream*3+2)
    bit=tl.full((),1,tl.uint64)<<(63-position)
    qword=qword^tl.where(is_key,tl.full((),0,tl.uint64),bit)
    kword=kword^tl.where(is_key,bit,tl.full((),0,tl.uint64))
    sign,_=_retrieve(qword,kword,vword,T,N)
    i=tl.arange(0,N);offsets=batch*T*C+i*C+channel
    credit=tl.sum((sign-tl.load(Y+offsets,i<T,other=0))*tl.load(DY+offsets,i<T,other=0),axis=0)
    offset=batch*T*C+position*C+channel
    logit=tl.where(is_key,tl.load(K+offset),tl.load(Q+offset))
    probability=tl.sigmoid(logit)
    gradient=credit*tl.where(logit>0,-1.,1.)*probability*(1-probability)
    tl.store(DQ+offset,gradient,mask=~is_key)
    tl.store(DK+offset,gradient,mask=is_key)


@tr.jit
def _backward_v(V,R,DY,DV,T:tl.constexpr,C:tl.constexpr,N:tl.constexpr):
    stream=tl.program_id(0);batch=stream//C;channel=stream%C
    i=tl.arange(0,N);offsets=batch*T*C+i*C+channel
    routes=tl.load(R+offsets,i<T,other=-1)
    dy=tl.load(DY+offsets,i<T,other=0)
    gradient=tl.sum(tl.where(i[:,None]==routes[None,:],2*dy[None,:],0.),axis=1)
    probability=tl.sigmoid(tl.load(V+offsets,i<T,other=0))
    tl.store(DV+offsets,gradient*probability*(1-probability),i<T)


def reference_symbols(q,k,v):
    """Simple exhaustive longest-suffix/latest-endpoint definition for one stream."""
    out=[]
    for t in range(len(q)):
        answer=-1
        found=False
        for length in range(t+1,0,-1):
            query=q[t+1-length:t+1]
            for end in range(t-1,length-2,-1):
                if k[end+1-length:end+1]==query:
                    answer=1 if v[end+1] else -1
                    found=True
                    break
            if found:break
        out.append(answer)
    return out


def reference_forward(q,k,v):
    bits=[x.detach().cpu()>0 for x in (q,k,v)]
    out=torch.empty_like(q,device='cpu')
    for b in range(q.shape[0]):
        for c in range(q.shape[2]):
            out[b,:,c]=torch.tensor(reference_symbols(*(x[b,:,c].tolist() for x in bits)))
    return out.to(q.device)


def reference_credits(q,k,v,dy):
    """Brute force finite flips plus the declared Bernoulli logit mapping."""
    bits=[x.detach().cpu()>0 for x in (q,k,v)]
    dy=dy.detach().cpu()
    grads=[torch.zeros_like(x,device='cpu') for x in (q,k,v)]
    for b in range(q.shape[0]):
        for c in range(q.shape[2]):
            streams=[x[b,:,c].tolist() for x in bits]
            base=torch.tensor(reference_symbols(*streams))
            for kind in range(3):
                for pos in range(q.shape[1]):
                    old=streams[kind][pos]
                    streams[kind][pos]=not old
                    alternate=torch.tensor(reference_symbols(*streams))
                    streams[kind][pos]=old
                    grads[kind][b,pos,c]=((alternate-base)*dy[b,:,c]).sum()*(1-2*int(old))
    return tuple(g.to(x.device)*x.sigmoid()*(1-x.sigmoid()) for g,x in zip(grads,(q,k,v)))


class HardROSA(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v):
        assert q.shape==k.shape==v.shape and q.ndim==3 and 1<=q.shape[1]<=64
        q,k,v=[x.contiguous() for x in (q,k,v)]
        if not q.is_cuda:
            y=reference_forward(q,k,v)
            ctx.save_for_backward(q,k,v,y)
            return y
        b,t,c=q.shape
        y=torch.empty_like(q);routes=torch.empty_like(q,dtype=torch.int32)
        packed=torch.empty((b*c,3),device=q.device,dtype=torch.uint64)
        _forward[(b*c,)](q,k,v,y,routes,packed,t,c,tr.next_power_of_2(t),num_warps=4)
        ctx.save_for_backward(q,k,v,y,routes,packed)
        return y

    @staticmethod
    def backward(ctx,dy):
        q,k,v,y,*rest=ctx.saved_tensors
        if not q.is_cuda:return reference_credits(q,k,v,dy)
        b,t,c=q.shape;dy=dy.contiguous()
        dq,dk,dv=[torch.empty_like(q) for _ in range(3)]
        _backward_qk[(b*c,2*t)](q,k,rest[1],y,dy,dq,dk,t,c,tr.next_power_of_2(t),num_warps=4)
        _backward_v[(b*c,)](v,rest[0],dy,dv,t,c,tr.next_power_of_2(t),num_warps=4)
        return dq,dk,dv


def rosa(q,k,v):return HardROSA.apply(q,k,v)
