"""KS3 fused W8A8 inference: INT8 dots, Horner evaluation and tensor parallelism.

Standalone FFN use (requires PyTorch, Triton and a supported GPU):

    import torch
    from ks3 import KS3
    from ks3_int8 import KS3Int8

    trained_cell = KS3(96, num_subspaces=368).cuda().eval()
    # Load trained_cell's weights before conversion.
    quantized_cell = KS3Int8(trained_cell)
    with torch.inference_mode():
        output = quantized_cell(torch.randn(2, 64, 96, device="cuda"))

Weights, activations and spine coefficients are INT8; matrix dots accumulate
in INT32. The input projection fuses Horner geometry, coordinate bypass,
ReLU-squared and activation requantization. Nonlinearities default to FP16;
scales and outputs are FP32. This is an inference-only implementation.

The tested configuration has 368 rank-1 charts, partitioned into four groups
of 92. TP1/TP2/TP4 use the same quantization groups. For TP, construct with
rank/world and initialize torch.distributed, or supply a sum reducer.

The optional full-RWKV convert() helper needs the repository's attention
adapter and keeps recurrent state FP32. Standalone KS3Int8 needs neither
that adapter nor the experiment/model-builder modules.
"""
import copy
import math
import torch
from torch import nn
import torch.distributed as dist
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice


@tr.jit
def _quant(X,Q,S,M:tl.constexpr,K:tl.constexpr,G:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);group=tl.program_id(1);j=tl.arange(0,B)
    x=tl.load(X+row*K+group*G+j,(group*G+j<K)&(j<G),other=0.)
    scale=tl.maximum(tl.max(tl.abs(x),0)/127.,1e-12)
    q=libdevice.nearbyint(x/scale).to(tl.int8)
    tl.store(Q+row*K+group*G+j,q,(group*G+j<K)&(j<G))
    tl.store(S+row*tr.cdiv(K,G)+group,scale)


@tr.jit
def _round_div127(value):
    # Triton integer // truncates toward zero. Correct both signs explicitly.
    quotient=value//127
    remainder=value-quotient*127
    return quotient+tl.where(remainder>63,1,tl.where(remainder < -63,-1,0))


@tr.jit
def _gemm(A,AS,W,WS,Y,YS,P,PS,SC,SH,
          LUT,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,G:tl.constexpr,
          BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,FUSED:tl.constexpr,NL:tl.constexpr,ACT:tl.constexpr):
    mi=tl.program_id(0)*BM+tl.arange(0,BM)
    # Fused output blocks are exactly one 92-feature quantization group.
    ni=tl.program_id(1)*(92 if FUSED else BN)+tl.arange(0,BN)
    ki=tl.arange(0,BK);NG=tr.cdiv(K,G)
    result=tl.full((BM,BN),0.,tl.float32)
    for group in range(NG):
        kk=group*G+ki
        av=tl.load(A+mi[:,None]*K+kk[None,:],(mi[:,None]<M)&(kk[None,:]<K)&(ki[None,:]<G),other=0)
        wv=tl.load(W+ni[None,:]*K+kk[:,None],(ni[None,:]<N)&(kk[:,None]<K)&(ki[:,None]<G)&((tl.arange(0,BN)[None,:]<92) if FUSED else True),other=0)
        acc=tl.dot(av,wv,out_dtype=tl.int32)
        sa=tl.load(AS+mi*NG+group,mi<M,other=0)
        sw=tl.load(WS+ni*NG+group,ni<N,other=0)
        result+=acc.to(tl.float32)*sa[:,None]*sw[None,:]
    if FUSED:
        valid=(ni<N)&(tl.arange(0,BN)<92)
        x=tl.minimum(tl.maximum(result,-65504.),65504.)
        c0=tl.load(P+ni*6+0,valid,other=0).to(tl.float32)
        c1=tl.load(P+ni*6+1,valid,other=0).to(tl.float32)
        c2=tl.load(P+ni*6+2,valid,other=0).to(tl.float32)
        c3=tl.load(P+ni*6+3,valid,other=0).to(tl.float32)
        c4=tl.load(P+ni*6+4,valid,other=0).to(tl.float32)
        c5=tl.load(P+ni*6+5,valid,other=0).to(tl.float32)
        ps=tl.load(PS+ni,valid,other=0)
        scale=tl.load(SC+ni,valid,other=1);shift=tl.load(SH+ni,valid,other=0)
        a=2*c0-2*c1+c2+c3;b=-3*c0+3*c1-2*c2-c3
        if NL==2:
            # INT8 tanh inputs/outputs, INT32 fixed-point Horner accumulators.
            # Q6 guard bits keep worst-case polynomial products within INT32.
            ti=libdevice.nearbyint(tl.minimum(tl.maximum(x*.25,-3.2),3.2)*(127./3.2)).to(tl.int32)
            tq=tl.load(LUT+ti+128).to(tl.int32)
            warp=scale[None,:]*(tq.to(tl.float32)*(4./127.)+shift[None,:])
            zi=libdevice.nearbyint(tl.minimum(tl.maximum(warp,-3.2),3.2)*(127./3.2)).to(tl.int32)
            zq=tl.load(LUT+zi+128).to(tl.int32)
            poly=a[None,:].to(tl.int32)*64
            poly=_round_div127(poly*tq*4)+b[None,:].to(tl.int32)*64
            poly=_round_div127(poly*tq*4)+c2[None,:].to(tl.int32)*64
            poly=_round_div127(poly*tq*4)+c0[None,:].to(tl.int32)*64
            warped=4*c5[None,:].to(tl.int32)*64
            warped=_round_div127(warped*zq)+2*c4[None,:].to(tl.int32)*64
            warped=_round_div127(warped*zq)-3*c5[None,:].to(tl.int32)*64
            warped=_round_div127(warped*zq)-c4[None,:].to(tl.int32)*64
            h=x+(poly+warped).to(tl.float32)*(ps[None,:]/64.)
            h=tl.where(valid[None,:],h,0.)
            pre_scale=tl.maximum(tl.max(tl.abs(h),1)/127.,1e-12)
            hq=libdevice.nearbyint(h/pre_scale[:,None]).to(tl.int32)
            hq=tl.maximum(hq,0);h=(hq*hq).to(tl.float32)*(pre_scale[:,None]*pre_scale[:,None])
        elif NL==1:
            th=(x*.25).to(tl.float16)
            t=(4*libdevice.tanh(th.to(tl.float32))).to(tl.float16)
            warp=(scale[None,:]*(t.to(tl.float32)+shift[None,:])).to(tl.float16)
            z=libdevice.tanh(warp.to(tl.float32)).to(tl.float16)
            # Each polynomial stage rounds to FP16; tanh uses FP32 internally.
            poly=(a[None,:].to(tl.float16)*t+b[None,:].to(tl.float16)).to(tl.float16)
            poly=(poly*t+c2[None,:].to(tl.float16)).to(tl.float16)
            poly=(poly*t+c0[None,:].to(tl.float16)).to(tl.float16)
            warped=(4*c5[None,:].to(tl.float16)*z+2*c4[None,:].to(tl.float16)).to(tl.float16)
            warped=(warped*z-3*c5[None,:].to(tl.float16)).to(tl.float16)
            warped=(warped*z-c4[None,:].to(tl.float16)).to(tl.float16)
            h=tl.maximum((x+(poly.to(tl.float32)+warped.to(tl.float32))*ps[None,:]).to(tl.float16),0.)
            h=(h*h).to(tl.float16).to(tl.float32)
        else:
            t=4*libdevice.tanh(x*.25);z=libdevice.tanh(scale[None,:]*(t+shift[None,:]))
            poly=((a[None,:]*t+b[None,:])*t+c2[None,:])*t+c0[None,:]
            warped=((4*c5[None,:]*z+2*c4[None,:])*z-3*c5[None,:])*z-c4[None,:]
            h=tl.maximum(x+(poly+warped)*ps[None,:],0.);h=h*h
        h=tl.where(valid[None,:],h,0.)
        hs=tl.maximum(tl.max(tl.abs(h),1)/127.,1e-12)
        q=libdevice.nearbyint(h/hs[:,None]).to(tl.int8)
        tl.store(Y+mi[:,None]*N+ni[None,:],q,(mi[:,None]<M)&valid[None,:])
        tl.store(YS+mi*tr.cdiv(N,92)+tl.program_id(1),hs,mi<M)
    elif ACT>0:
        if ACT==2:result=libdevice.tanh(result)
        elif ACT==3:result=tl.sigmoid(result)
        result=tl.where(ni[None,:]<N,result,0.)
        scale=tl.maximum(tl.max(tl.abs(result),1)/127.,1e-12)
        q=libdevice.nearbyint(result/scale[:,None]).to(tl.int8)
        tl.store(Y+mi[:,None]*N+ni[None,:],q,(mi[:,None]<M)&(ni[None,:]<N))
        tl.store(YS+mi,scale,mi<M)
    else:
        tl.store(Y+mi[:,None]*N+ni[None,:],result,(mi[:,None]<M)&(ni[None,:]<N))


def quantize(x,group=128):
    x=x.contiguous();k=x.shape[-1];m=x.numel()//k
    q=torch.empty_like(x,dtype=torch.int8);s=torch.empty((m,tr.cdiv(k,group)),device=x.device)
    _quant[(m,tr.cdiv(k,group))](x,q,s,m,k,group,tr.next_power_of_2(group),num_warps=4)
    return q,s


@torch.no_grad()
def pack_weight(w,group):
    n,k=w.shape;pad=(-k)%group
    x=torch.nn.functional.pad(w.float(),(0,pad)).reshape(n,-1,group)
    scales=x.abs().amax(-1).clamp_min(1e-10)/127.
    q=(x/scales[...,None]).round().clamp(-127,127).to(torch.int8).reshape(n,-1)[:,:k].contiguous()
    return q,scales.contiguous()


class Int8Linear(nn.Module):
    def __init__(self,weight,group=128):
        super().__init__();q,s=pack_weight(weight,group)
        self.register_buffer('qweight',q);self.register_buffer('scales',s)
        self.in_features=weight.shape[1];self.out_features=weight.shape[0];self.group=group
    def from_quantized(self,q,s):
        shape=q.shape[:-1];m=q.numel()//self.in_features
        y=torch.empty((*shape,self.out_features),device=q.device,dtype=torch.float32)
        _gemm[(tr.cdiv(m,16),tr.cdiv(self.out_features,32))](q,s,self.qweight,self.scales,y,y,y,y,y,y,y,m,self.out_features,self.in_features,self.group,16,32,tr.next_power_of_2(self.group),False,0,0,num_warps=4)
        return y
    def forward(self,x):return self.from_quantized(*quantize(x,self.group))


class Int8Controller(nn.Module):
    """Fuse controller input GEMM, activation and next-GEMM quantization."""
    def __init__(self,base):
        super().__init__();self.down=Int8Linear(base.down.weight);self.up=Int8Linear(base.up.weight)
        self.activation={'linear':1,'tanh':2,'sigmoid':3}[base.activation]
    def forward(self,x):
        q,s=quantize(x);m=x.numel()//x.shape[-1];n=self.down.out_features;k=x.shape[-1]
        h=torch.empty((*x.shape[:-1],n),device=x.device,dtype=torch.int8);hs=torch.empty((m,1),device=x.device)
        _gemm[(tr.cdiv(m,16),1)](q,s,self.down.qweight,self.down.scales,h,hs,h,h,h,h,h,m,n,k,128,16,max(32,tr.next_power_of_2(n)),128,False,0,self.activation,num_warps=4)
        return self.up.from_quantized(h,hs)


class Int8Embedding(nn.Module):
    def __init__(self,weight):
        super().__init__();q,s=pack_weight(weight,weight.shape[-1]);self.register_buffer('qweight',q);self.register_buffer('scales',s)
    def forward(self,tokens):return self.qweight[tokens].float()*self.scales[tokens]


class KS3Int8(nn.Module):
    """Pack a trained KS3 cell for fused inference, optionally sharded across GPUs.

    ``cell`` can be the standalone KS3 reference or the legacy rank-1 cell.
    Load its trained weights before constructing this module. The packed
    buffers support state_dict/save/load; conversion does not mutate the cell.
    """
    def __init__(self,cell,rank=0,world=1,nonlinear='fp16',reducer=None):
        assert nonlinear in ('fp32','fp16','int8')
        super().__init__();assert cell.subspace_rank==1 and cell.num_subspaces==368
        assert world in (1,2,4);local=368//world;start=rank*local;end=start+local
        self.down=Int8Linear(cell.down.weight[start:end],128)
        self.up=Int8Linear(cell.up.weight[:,start:end],92)
        if hasattr(cell, 'coefficients'):
            coefficients, raw_scale, raw_shift = cell.coefficients, cell.raw_scale, cell.raw_shift
        else:
            coefficients, raw_scale, raw_shift = cell.geo_spine, cell.geo_raw_scale, cell.geo_raw_shift
        spine=coefficients.detach().reshape(368,6)[start:end]
        ps=spine.abs().amax(-1).clamp_min(1e-10)/127.
        self.register_buffer('spine',(spine/ps[:,None]).round().clamp(-127,127).to(torch.int8))
        self.register_buffer('spine_scale',ps)
        self.register_buffer('warp_scale',torch.nn.functional.softplus(raw_scale.detach().reshape(-1)[start:end])+1e-4)
        self.register_buffer('warp_shift',.5*torch.tanh(raw_shift.detach().reshape(-1)[start:end]))
        self.world=world;self.local=local;self.reducer=reducer;self.nonlinear={'fp32':0,'fp16':1,'int8':2}[nonlinear]
        self.register_buffer('tanh_lut',(torch.tanh(torch.arange(-128,128,device=spine.device).float()*(3.2/127.))*127).round().clamp(-127,127).to(torch.int8))
    def forward(self,x):
        q,s=quantize(x,128);m=x.numel()//x.shape[-1]
        h=torch.empty((*x.shape[:-1],self.local),device=x.device,dtype=torch.int8)
        hs=torch.empty((m,self.local//92),device=x.device)
        _gemm[(tr.cdiv(m,16),self.local//92)](q,s,self.down.qweight,self.down.scales,h,hs,self.spine,self.spine_scale,self.warp_scale,self.warp_shift,self.tanh_lut,m,self.local,x.shape[-1],128,16,128,128,True,self.nonlinear,0,num_warps=4)
        y=self.up.from_quantized(h,hs)
        if self.world>1:
            if self.reducer is None:dist.all_reduce(y)
            else:y=self.reducer(y)
        return y


def convert(model,rank=0,world=1,nonlinear='fp16',reducer=None):
    """Export W8A8 maps, shard KS3 subspaces and assign the three RWKV heads."""
    model=copy.deepcopy(model).eval()
    from experiments.rwkv7_ks2.ks3_attention_tp import HeadParallelAttention
    for block in model.blocks:
        block.ffn=KS3Int8(block.ffn,rank,world,nonlinear,reducer)
        block.attention=HeadParallelAttention(block.attention,rank,world,reducer)
    def visit(parent):
        for name,child in list(parent.named_children()):
            if isinstance(child,(KS3Int8,Int8Linear,Int8Embedding,Int8Controller)):continue
            if child.__class__.__name__=='Controller':setattr(parent,name,Int8Controller(child))
            elif isinstance(child,nn.Linear):
                assert child.bias is None
                setattr(parent,name,Int8Linear(child.weight))
            elif isinstance(child,nn.Embedding):setattr(parent,name,Int8Embedding(child.weight))
            else:visit(child)
    visit(model)
    for p in model.parameters():p.requires_grad_(False)
    return model


@tr.jit
def _state_forward(R,D,K,V,N,A,Y,T:tl.constexpr,H:tl.constexpr,W:tl.constexpr):
    bh=tl.program_id(0);b=bh//H;h=bh%H;i=tl.arange(0,W)
    state=tl.full((W,W),0.,tl.float32)
    for pos in range(T):
        off=((b*T+pos)*H+h)*W+i
        r=tl.load(R+off);d=tl.load(D+off);k=tl.load(K+off);v=tl.load(V+off);n=tl.load(N+off);a=tl.load(A+off)
        removal=tl.sum(state*n[None,:],1)
        state=state*d[None,:]-removal[:,None]*(n*a)[None,:]+v[:,None]*k[None,:]
        tl.store(Y+off,tl.sum(state*r[None,:],1))


def inference_recurrence(*args):
    args=tuple(x.contiguous() for x in args);b,t,h,w=args[0].shape
    assert all(x.dtype==torch.float32 for x in args)
    y=torch.empty_like(args[0]);_state_forward[(b*h,)](*args,y,t,h,w,num_warps=4,enable_fp_fusion=False)
    return y


# Historical experiment scripts can keep their original import name.
Int8Rank1FFN = KS3Int8
