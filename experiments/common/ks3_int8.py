"""KS3 fused W8A8 forward and fused straight-through backward, with TP support.

Standalone FFN use (requires PyTorch, Triton and a supported GPU):

    import torch
    from experiments.common.ks3 import KS3
    from experiments.common.ks3_int8 import KS3Int8

    trained_cell = KS3(96, num_subspaces=368).cuda().eval()
    # Load trained_cell's weights before conversion.
    quantized_cell = KS3Int8(trained_cell)
    with torch.inference_mode():
        output = quantized_cell(torch.randn(2, 64, 96, device="cuda"))

Weights, activations and spine coefficients are INT8; matrix dots accumulate
in INT32. The input projection fuses Horner geometry, coordinate bypass,
ReLU-squared and activation requantization. Nonlinearities default to FP16;
all quantization scales are stored in FP16 with GS32 grouping. Outputs and
warp scalars remain FP32. A six-coefficient spine row uses a partial GS32 group.

For training, pass ``trainable=True`` to KS3Int8, then create the optimizer
from that module's parameters. It owns FP32 master copies and repacks them
on every forward. Backward fuses the output-projection VJP with the geometry
VJP, reduces all eight geometry gradients, and computes both projection
gradients and the input gradient. Packed operands are dequantized inside
the backward GEMMs; backward products and accumulators are FP32, not INT8.

Quantization uses an identity straight-through estimator with detached scales.
All modes differentiate smooth FP32 KS3 geometry at the quantized coordinates
and coefficients. This is also the surrogate for FP16 rounding and INT8 LUTs;
it is not the literal derivative of discrete rounding. Masters stay FP32;
inputs may be FP32, FP16 or BF16. Output follows autocast's dtype, or the input
dtype outside autocast. First-order gradients only. ``to_inference()`` packs
the updated masters into an inference-only module without optimizer state.

Input/output dimensions, number of charts, and chart rank are arbitrary
positive integers. Rank 1 fuses input GEMM and geometry; higher ranks add a
chart-mean reduction and a fused geometry/requantization kernel. TP partitions
whole charts and whole GS32 groups. The 368-rank-1 layout splits into
[96, 96, 96, 80] features on TP4. Tiny layouts may leave some ranks empty.
Initialize torch.distributed for TP, or supply a sum reducer.

FP16 scales are rounded before quantization, bounded to [2**-24, 65504], and
promoted to FP32 for arithmetic. Codes saturate to [-127,127]. Groups shorter
than 32 use a partial group; no learned padding parameters are introduced.

The optional full-RWKV convert() helper needs the repository's attention
adapter and keeps recurrent state FP32. Standalone KS3Int8 needs neither
that adapter nor the experiment/model-builder modules.

Use ``fuse_training(torch_ks3_cell)`` when an existing model/optimizer owns the
parameters: it preserves parameter identities, names and checkpoint keys.
``ks3_int8(...)`` is the functional equivalent for external parameter views.
"""
import copy
import math
import torch
from torch import nn
from torch.autograd.function import once_differentiable
import torch.distributed as dist
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice


GROUP_SIZE = 32
SCALE_DTYPE = torch.float16


def chart_range(charts, subspace_rank, rank=0, world=1):
    """Assign whole charts and whole GS32 groups; tiny layouts may have idle ranks."""
    if any(not isinstance(v,int) or isinstance(v,bool) or v < 1 for v in (charts,subspace_rank,world)) or not 0 <= rank < world:
        raise ValueError('Positive chart count, rank, world size and a valid TP rank are required.')
    charts_per_group = GROUP_SIZE // math.gcd(GROUP_SIZE,subspace_rank)
    blocks = tr.cdiv(charts,charts_per_group)
    first = rank*(blocks//world)+min(rank,blocks%world)
    count = blocks//world+int(rank < blocks%world)
    return min(charts,first*charts_per_group),min(charts,(first+count)*charts_per_group)


def _rounded_scale(maximum):
    return (maximum.float()/127.).clamp(2.**-24,65504.).to(SCALE_DTYPE)


@tr.jit
def _half_scale(maximum):
    # Round BEFORE quantizing so encoding and decoding use the same FP16 scale.
    return tl.minimum(tl.maximum(maximum/127., 2.**-24), 65504.).to(tl.float16).to(tl.float32)


@tr.jit
def _spine_features(x, mean, c0, c1, c2, c3, c4, c5, ps, scale, shift, LUT,
                    valid, NL: tl.constexpr):
    a=2*c0-2*c1+c2+c3;b=-3*c0+3*c1-2*c2-c3
    if NL==2:
        # INT8 tanh inputs/outputs, INT32 fixed-point Horner accumulators.
        # Q6 guard bits keep worst-case polynomial products within INT32.
        ti=libdevice.nearbyint(tl.minimum(tl.maximum(mean*.25,-3.2),3.2)*(127./3.2)).to(tl.int32)
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
        pre_scale=_half_scale(tl.max(tl.abs(h),1))
        hq=tl.minimum(tl.maximum(libdevice.nearbyint(h/pre_scale[:,None]),-127.),127.).to(tl.int32)
        hq=tl.maximum(hq,0);h=(hq*hq).to(tl.float32)*(pre_scale[:,None]*pre_scale[:,None])
    elif NL==1:
        th=(mean*.25).to(tl.float16)
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
        t=4*libdevice.tanh(mean*.25);z=libdevice.tanh(scale[None,:]*(t+shift[None,:]))
        poly=((a[None,:]*t+b[None,:])*t+c2[None,:])*t+c0[None,:]
        warped=((4*c5[None,:]*z+2*c4[None,:])*z-3*c5[None,:])*z-c4[None,:]
        h=tl.maximum(x+(poly+warped)*ps[None,:],0.);h=h*h
    return tl.where(valid[None,:],h,0.)


@tr.jit
def _quant_tile(X,Q,S,tile,M:tl.constexpr,K:tl.constexpr,G:tl.constexpr,
                B:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr):
    # Eight groups per program amortize launch/workgroup overhead. Each row
    # still has its own GS32 scale, including partial final groups.
    groups=tl.cdiv(K,G)
    index=tile*8+tl.arange(0,8)
    row=index//groups;group=index%groups;j=tl.arange(0,B)
    offsets=row[:,None]*K+group[:,None]*G+j[None,:]
    valid=(row[:,None]<M)&(group[:,None]*G+j[None,:]<K)&(j[None,:]<G)
    source=row[:,None]*S0+(group[:,None]*G+j[None,:])*S1
    x=tl.load(X+source,valid,other=0.).to(tl.float32)
    scale=_half_scale(tl.max(tl.abs(x),1))
    q=tl.minimum(tl.maximum(libdevice.nearbyint(x/scale[:,None]),-127.),127.).to(tl.int8)
    tl.store(Q+offsets,q,valid)
    tl.store(S+index,scale,row<M)


@tr.jit
def _quant(X,Q,S,M:tl.constexpr,K:tl.constexpr,G:tl.constexpr,B:tl.constexpr):
    _quant_tile(X,Q,S,tl.program_id(0),M,K,G,B,K,1)


@tr.jit
def _pack_training(X,D,U,P,RS,RH,QX,SX,QD,SD,QU,SU,QP,SP,SC,SH,
                   M:tl.constexpr,I:tl.constexpr,O:tl.constexpr,N:tl.constexpr,
                   R:tl.constexpr,D0:tl.constexpr,D1:tl.constexpr,
                   U0:tl.constexpr,U1:tl.constexpr):
    """Pack current activations, both projections and geometry in one launch."""
    pid=tl.program_id(0)
    tx=tl.cdiv(M*tl.cdiv(I,32),8)
    td=tl.cdiv(N*tl.cdiv(I,32),8)
    tu=tl.cdiv(O*tl.cdiv(N,32),8)
    if pid<tx:
        _quant_tile(X,QX,SX,pid,M,I,32,32,I,1)
    elif pid<tx+td:
        _quant_tile(D,QD,SD,pid-tx,N,I,32,32,D0,D1)
    elif pid<tx+td+tu:
        _quant_tile(U,QU,SU,pid-tx-td,O,N,32,32,U0,U1)
    else:
        tile=pid-tx-td-tu
        _quant_tile(P,QP,SP,tile,N,6,32,8,6,1)
        coordinate=tile*8+tl.arange(0,8)
        chart=coordinate//R
        valid=(coordinate<N)&(coordinate%R==0)
        raw=tl.load(RS+chart,valid,other=0.)
        shift=tl.load(RH+chart,valid,other=0.)
        scale=tl.where(raw>20.,raw,libdevice.log1p(libdevice.exp(raw)))+1e-4
        tl.store(SC+chart,scale,valid)
        tl.store(SH+chart,.5*libdevice.tanh(shift),valid)


def _pack_training_tensors(x,down,up,coefficients,raw_scale,raw_shift,rank):
    m,i=x.shape;n=down.shape[0];o=up.shape[0]
    def buffers(rows,cols):
        return (torch.empty((rows,cols),device=x.device,dtype=torch.int8),
                torch.empty((rows,tr.cdiv(cols,32)),device=x.device,dtype=SCALE_DTYPE))
    qx,sx=buffers(m,i);qd,sd=buffers(n,i);qu,su=buffers(o,n);p,ps=buffers(n,6)
    scale=torch.empty_like(raw_scale);shift=torch.empty_like(raw_shift)
    tiles=sum(tr.cdiv(rows*tr.cdiv(cols,32),8) for rows,cols in ((m,i),(n,i),(o,n),(n,6)))
    _pack_training[(tiles,)](x,down,up,coefficients.reshape(-1,6).contiguous(),raw_scale,raw_shift,
        qx,sx,qd,sd,qu,su,p,ps,scale,shift,m,i,o,n,rank,*down.stride(),*up.stride(),num_warps=4)
    return qx,sx,qd,sd,qu,su,p,ps.flatten(),scale,shift


@tr.jit
def _round_div127(value):
    # Triton integer // truncates toward zero. Correct both signs explicitly.
    quotient=value//127
    remainder=value-quotient*127
    return quotient+tl.where(remainder>63,1,tl.where(remainder < -63,-1,0))


@tr.jit
def _gemm(A,AS,W,WS,Y,YS,P,PS,SC,SH,
          LUT,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,G:tl.constexpr,
          BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,FUSED:tl.constexpr,NL:tl.constexpr,ACT:tl.constexpr,
          COORD=None,SAVE_COORD:tl.constexpr=False):
    tl.static_assert(not (FUSED or ACT>0) or BN==32, "GS32 fused epilogues require BN=32")
    mi=tl.program_id(0)*BM+tl.arange(0,BM)
    # Fused output blocks are exactly one GS32 quantization group.
    ni=tl.program_id(1)*BN+tl.arange(0,BN)
    ki=tl.arange(0,BK);NG=tl.cdiv(K,G)
    result=tl.full((BM,BN),0.,tl.float32)
    for group in range(NG):
        kk=group*G+ki
        av=tl.load(A+mi[:,None]*K+kk[None,:],(mi[:,None]<M)&(kk[None,:]<K)&(ki[None,:]<G),other=0)
        wv=tl.load(W+ni[None,:]*K+kk[:,None],(ni[None,:]<N)&(kk[:,None]<K)&(ki[:,None]<G),other=0)
        acc=tl.dot(av,wv,out_dtype=tl.int32)
        sa=tl.load(AS+mi*NG+group,mi<M,other=0).to(tl.float32)
        sw=tl.load(WS+ni*NG+group,ni<N,other=0).to(tl.float32)
        result+=acc.to(tl.float32)*sa[:,None]*sw[None,:]
    if FUSED:
        valid=ni<N
        if SAVE_COORD:
            tl.store(COORD+mi[:,None]*N+ni[None,:],result,(mi[:,None]<M)&valid[None,:])
        x=tl.minimum(tl.maximum(result,-65504.),65504.)
        c0=tl.load(P+ni*6+0,valid,other=0).to(tl.float32)
        c1=tl.load(P+ni*6+1,valid,other=0).to(tl.float32)
        c2=tl.load(P+ni*6+2,valid,other=0).to(tl.float32)
        c3=tl.load(P+ni*6+3,valid,other=0).to(tl.float32)
        c4=tl.load(P+ni*6+4,valid,other=0).to(tl.float32)
        c5=tl.load(P+ni*6+5,valid,other=0).to(tl.float32)
        ps=tl.load(PS+ni,valid,other=0).to(tl.float32)
        scale=tl.load(SC+ni,valid,other=1);shift=tl.load(SH+ni,valid,other=0)
        h=_spine_features(x,x,c0,c1,c2,c3,c4,c5,ps,scale,shift,LUT,valid,NL)
        h=tl.where(valid[None,:],h,0.)
        hs=_half_scale(tl.max(tl.abs(h),1))
        q=tl.minimum(tl.maximum(libdevice.nearbyint(h/hs[:,None]),-127.),127.).to(tl.int8)
        tl.store(Y+mi[:,None]*N+ni[None,:],q,(mi[:,None]<M)&valid[None,:])
        tl.store(YS+mi*tl.cdiv(N,32)+tl.program_id(1),hs,mi<M)
    elif ACT>0:
        if ACT==2:result=libdevice.tanh(result)
        elif ACT==3:result=tl.sigmoid(result)
        result=tl.where(ni[None,:]<N,result,0.)
        scale=_half_scale(tl.max(tl.abs(result),1))
        q=tl.minimum(tl.maximum(libdevice.nearbyint(result/scale[:,None]),-127.),127.).to(tl.int8)
        tl.store(Y+mi[:,None]*N+ni[None,:],q,(mi[:,None]<M)&(ni[None,:]<N))
        tl.store(YS+mi*tl.cdiv(N,32)+tl.program_id(1),scale,mi<M)
    else:
        tl.store(Y+mi[:,None]*N+ni[None,:],result,(mi[:,None]<M)&(ni[None,:]<N))


def quantize(x,group=32):
    x=x.contiguous();k=x.shape[-1];m=math.prod(x.shape[:-1])
    q=torch.empty_like(x,dtype=torch.int8);s=torch.empty((m,tr.cdiv(k,group)),device=x.device,dtype=SCALE_DTYPE)
    if m and k: _quant[(tr.cdiv(m*tr.cdiv(k,group),8),)](x,q,s,m,k,group,tr.next_power_of_2(group),num_warps=4)
    return q,s


@torch.no_grad()
def pack_weight(w,group=32):
    if w.is_cuda:
        return quantize(w,group)
    n,k=w.shape;pad=(-k)%group
    x=torch.nn.functional.pad(w.float(),(0,pad)).reshape(n,tr.cdiv(k,group),group)
    scales=_rounded_scale(x.abs().amax(-1))
    q=(x/scales.float()[...,None]).round().clamp(-127,127).to(torch.int8).reshape(n,tr.cdiv(k,group)*group)[:,:k].contiguous()
    return q,scales.contiguous()


class Int8Linear(nn.Module):
    def __init__(self,weight,group=32):
        super().__init__();q,s=pack_weight(weight,group)
        self.register_buffer('qweight',q);self.register_buffer('scales',s)
        self.in_features=weight.shape[1];self.out_features=weight.shape[0];self.group=group
    def from_quantized(self,q,s):
        shape=q.shape[:-1];m=math.prod(shape)
        y=torch.empty((*shape,self.out_features),device=q.device,dtype=torch.float32)
        if not self.in_features: return y.zero_()
        if not m: return y
        _gemm[(tr.cdiv(m,16),tr.cdiv(self.out_features,32))](q,s,self.qweight,self.scales,y,y,y,y,y,y,y,m,self.out_features,self.in_features,self.group,16,32,tr.next_power_of_2(self.group),False,0,0,COORD=y,SAVE_COORD=False,num_warps=4)
        return y
    def forward(self,x):return self.from_quantized(*quantize(x,self.group))


class Int8Controller(nn.Module):
    """Fuse controller input GEMM, activation and next-GEMM quantization."""
    def __init__(self,base):
        super().__init__();self.down=Int8Linear(base.down.weight);self.up=Int8Linear(base.up.weight)
        self.activation={'linear':1,'tanh':2,'sigmoid':3}[base.activation]
    def forward(self,x):
        q,s=quantize(x);m=x.numel()//x.shape[-1];n=self.down.out_features;k=x.shape[-1]
        h=torch.empty((*x.shape[:-1],n),device=x.device,dtype=torch.int8);hs=torch.empty((m,tr.cdiv(n,32)),device=x.device,dtype=SCALE_DTYPE)
        _gemm[(tr.cdiv(m,16),tr.cdiv(n,32))](q,s,self.down.qweight,self.down.scales,h,hs,h,h,h,h,h,m,n,k,32,16,32,32,False,0,self.activation,COORD=h,SAVE_COORD=False,num_warps=4)
        return self.up.from_quantized(h,hs)


class Int8Embedding(nn.Module):
    def __init__(self,weight):
        super().__init__();q,s=pack_weight(weight,32);self.register_buffer('qweight',q);self.register_buffer('scales',s)
    def forward(self,tokens):return self.qweight[tokens].float()*self.scales[tokens].float().repeat_interleave(32,-1)[...,:self.qweight.shape[1]]


@tr.jit
def _geometry_quant(C, MEAN, P, PS, SC, SH, LUT, H, HS,
                    M:tl.constexpr, N:tl.constexpr, RANK:tl.constexpr,
                    NL:tl.constexpr, BM:tl.constexpr=16, BN:tl.constexpr=32):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    valid=cols<N
    original=tl.load(C+rows[:,None]*N+cols[None,:],(rows[:,None]<M)&valid[None,:],other=0.)
    x=tl.minimum(tl.maximum(original,-65504.),65504.)
    mean=tl.load(MEAN+rows[:,None]*(N//RANK)+cols[None,:]//RANK,
                 (rows[:,None]<M)&valid[None,:],other=0.)
    c0=tl.load(P+cols*6,valid,other=0).to(tl.float32)
    c1=tl.load(P+cols*6+1,valid,other=0).to(tl.float32)
    c2=tl.load(P+cols*6+2,valid,other=0).to(tl.float32)
    c3=tl.load(P+cols*6+3,valid,other=0).to(tl.float32)
    c4=tl.load(P+cols*6+4,valid,other=0).to(tl.float32)
    c5=tl.load(P+cols*6+5,valid,other=0).to(tl.float32)
    ps=tl.load(PS+cols,valid,other=0).to(tl.float32)
    scale=tl.load(SC+cols//RANK,valid,other=1).to(tl.float32)
    shift=tl.load(SH+cols//RANK,valid,other=0).to(tl.float32)
    h=_spine_features(x,mean,c0,c1,c2,c3,c4,c5,ps,scale,shift,LUT,valid,NL)
    hs=_half_scale(tl.max(tl.abs(h),1))
    q=tl.minimum(tl.maximum(libdevice.nearbyint(h/hs[:,None]),-127.),127.).to(tl.int8)
    tl.store(H+rows[:,None]*N+cols[None,:],q,(rows[:,None]<M)&valid[None,:])
    tl.store(HS+rows*tl.cdiv(N,32)+tl.program_id(1),hs,rows<M)


def pack_spine(coefficients):
    # Six coefficients per output coordinate form a partial GS32 row.
    q, scales=pack_weight(coefficients.reshape(-1,6),32)
    return q, scales.flatten()


def _project_features(qx,sx,qd,sd,p,ps,scale,shift,lut,subspace_rank,nonlinear,save=False):
    m=math.prod(qx.shape[:-1]);n,k=qd.shape
    h=torch.empty((m,n),device=qx.device,dtype=torch.int8)
    hs=torch.empty((m,tr.cdiv(n,32)),device=qx.device,dtype=SCALE_DTYPE)
    coords=torch.empty((m,n),device=qx.device,dtype=torch.float32) if save or subspace_rank>1 else None
    if not m or not n:
        return h,hs,coords,None
    if subspace_rank==1:
        _gemm[(tr.cdiv(m,16),tr.cdiv(n,32))](
            qx,sx,qd,sd,h,hs,p,ps,scale,shift,lut,m,n,k,32,16,32,32,True,nonlinear,0,
            COORD=coords,SAVE_COORD=save,num_warps=4)
        mean=coords
    else:
        _gemm[(tr.cdiv(m,16),tr.cdiv(n,32))](
            qx,sx,qd,sd,coords,coords,coords,coords,coords,coords,coords,
            m,n,k,32,16,32,32,False,0,0,COORD=coords,SAVE_COORD=False,num_warps=4)
        mean=coords.clamp(-65504.,65504.).reshape(m,n//subspace_rank,subspace_rank).mean(-1)
        _geometry_quant[(tr.cdiv(m,16),tr.cdiv(n,32))](
            coords,mean,p,ps,scale,shift,lut,h,hs,m,n,subspace_rank,nonlinear,BM=16,BN=32,num_warps=4)
    return h,hs,coords,mean


class KS3Int8(nn.Module):
    """GS32/FP16-scale KS3 inference or training with arbitrary widths and chart rank.

    ``cell`` supplies positive input/output widths, num_subspaces and subspace_rank.
    With ``trainable=True``, optimize this module's FP32 ``master_*`` parameters.
    ``eval()`` keeps those masters current; ``to_inference()`` freezes them.

    TP assigns whole charts and whole quantization groups. Replicated input and
    upstream gradients are required; only output/input partials are all-reduced.
    Small configurations can leave ranks without a chart; they still participate
    in collectives. The optional full-model convert() helper targets the historical
    RWKV harness; this FFN itself is independent of its width/head restrictions.
    """
    def __init__(self,cell,rank=0,world=1,nonlinear='fp16',reducer=None,*,trainable=False):
        super().__init__()
        if nonlinear not in ('fp32','fp16','int8'):
            raise ValueError('nonlinear must be fp32, fp16 or int8')
        self.num_subspaces=cell.num_subspaces
        self.subspace_rank=cell.subspace_rank
        self.chart_start,self.chart_end=chart_range(self.num_subspaces,self.subspace_rank,rank,world)
        start,end=self.chart_start*self.subspace_rank,self.chart_end*self.subspace_rank
        self.local_subspaces=self.chart_end-self.chart_start
        self.local=end-start
        self.in_features=cell.down.weight.shape[1];self.out_features=cell.up.weight.shape[0]
        features=self.num_subspaces*self.subspace_rank
        if cell.down.weight.shape[0]!=features or cell.up.weight.shape[1]!=features:
            raise ValueError('Projection shapes do not match subspace count and rank')
        if hasattr(cell,'coefficients'):
            coefficients=cell.coefficients.reshape(self.num_subspaces,self.subspace_rank,6)
            raw_scale,raw_shift=cell.raw_scale,cell.raw_shift
        else:
            coefficients=cell.geo_spine.permute(0,2,1)
            raw_scale,raw_shift=cell.geo_raw_scale,cell.geo_raw_shift
        coeff=coefficients.detach()[self.chart_start:self.chart_end].contiguous()
        if self.subspace_rank==1:coeff=coeff.reshape(self.local_subspaces,6)
        raw_scale=raw_scale.detach().reshape(self.num_subspaces)[self.chart_start:self.chart_end]
        raw_shift=raw_shift.detach().reshape(self.num_subspaces)[self.chart_start:self.chart_end]
        down=cell.down.weight.detach()[start:end];up=cell.up.weight.detach()[:,start:end]
        self.down=Int8Linear(down);self.up=Int8Linear(up)
        spine,ps=pack_spine(coeff)
        self.register_buffer('spine',spine);self.register_buffer('spine_scale',ps)
        self.register_buffer('warp_scale',torch.nn.functional.softplus(raw_scale.float())+1e-4)
        self.register_buffer('warp_shift',.5*torch.tanh(raw_shift.float()))
        self.world=world;self.tp_rank=rank;self.reducer=reducer
        self.nonlinear={'fp32':0,'fp16':1,'int8':2}[nonlinear]
        self.register_buffer('tanh_lut',(torch.tanh(torch.arange(-128,128,device=spine.device).float()*(3.2/127.))*127).round().clamp(-127,127).to(torch.int8))
        self.trainable=trainable
        if trainable:
            for name,value in (('down',down),('up',up),('coefficients',coeff),
                               ('raw_scale',raw_scale),('raw_shift',raw_shift)):
                setattr(self,'master_'+name,nn.Parameter(value.float().clone()))

    def forward(self,x):
        if x.shape[-1]!=self.in_features:
            raise ValueError('Input feature width does not match this KS3 cell')
        if self.trainable:
            return _KS3Train.apply(x,self.master_down,self.master_up,self.master_coefficients,
                self.master_raw_scale,self.master_raw_shift,self.tanh_lut,
                self.nonlinear,self.world,self.reducer,self.subspace_rank)
        q,s=quantize(x)
        h,hs,_,_=_project_features(q,s,self.down.qweight,self.down.scales,self.spine,
            self.spine_scale,self.warp_scale,self.warp_shift,self.tanh_lut,self.subspace_rank,self.nonlinear)
        y=self.up.from_quantized(h,hs).reshape(*x.shape[:-1],self.out_features)
        if self.world>1 and y.numel():
            if self.reducer is None:dist.all_reduce(y)
            else:y=self.reducer(y)
        output_dtype=torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else x.dtype
        return y.to(output_dtype)

    @torch.no_grad()
    def to_inference(self):
        """Return a frozen copy, packing the latest master weights on each rank."""
        result=copy.deepcopy(self)
        if not self.trainable:return result.eval()
        result.down=Int8Linear(self.master_down);result.up=Int8Linear(self.master_up)
        result.spine,result.spine_scale=pack_spine(self.master_coefficients)
        result.warp_scale=torch.nn.functional.softplus(self.master_raw_scale)+1e-4
        result.warp_shift=.5*torch.tanh(self.master_raw_shift)
        for name in ('down','up','coefficients','raw_scale','raw_shift'):
            delattr(result,'master_'+name)
        result.trainable=False
        return result.eval()


@tr.jit
def _gradient_dot(A,B,BS,row_tile,col_tile,
                  M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                  A0:tl.constexpr,A1:tl.constexpr,GROUP:tl.constexpr,
                  BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    rows = row_tile * BM + tl.arange(0, BM)
    cols = col_tile * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    value = tl.full((BM, BN), 0., tl.float32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + inner
        a = tl.load(A + rows[:, None] * A0 + kk[None, :] * A1,
                    (rows[:, None] < M) & (kk[None, :] < K), other=0)
        b = tl.load(B + kk[:, None] * N + cols[None, :],
                    (kk[:, None] < K) & (cols[None, :] < N), other=0)
        bs = tl.load(BS + kk[:, None] * tl.cdiv(N, GROUP) + cols[None, :] // GROUP,
                     (kk[:, None] < K) & (cols[None, :] < N), other=0)
        value += tl.dot(a.to(tl.float32), b.to(tl.float32) * bs,
                        input_precision="ieee")
    return value


@tr.jit
def _backward_mm(A, B, BS, OUT, COORD, P, PS, SCALE, SHIFT, RAW_SCALE,
                 RAW_SHIFT, PARTIAL, MEAN, DMEAN, RANK: tl.constexpr,
                 M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                 A0: tl.constexpr, A1: tl.constexpr, GROUP: tl.constexpr,
                 GEOMETRY: tl.constexpr,
                 BM: tl.constexpr = 16, BN: tl.constexpr = 32,
                 BK: tl.constexpr = 32):
    """FP32 VJP GEMM with tiled INT8 dequantization and optional KS3 epilogue.

    B is a row-major K x N packed tensor, grouped along N. A may be a
    transpose view. The geometry epilogue avoids materializing d(features).
    """
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    value = _gradient_dot(A,B,BS,tl.program_id(0),tl.program_id(1),
                          M,N,K,A0,A1,GROUP,BM,BN,BK)

    if GEOMETRY:
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        original = tl.load(COORD + rows[:, None] * N + cols[None, :], mask, other=0)
        x = tl.minimum(tl.maximum(original, -65504.), 65504.)
        scale = tl.load(SCALE + cols//RANK, cols < N, other=1).to(tl.float32)
        shift = tl.load(SHIFT + cols//RANK, cols < N, other=0).to(tl.float32)
        ps = tl.load(PS + cols, cols < N, other=0).to(tl.float32)
        c0 = tl.load(P + cols * 6, cols < N, other=0).to(tl.float32) * ps
        c1 = tl.load(P + cols * 6 + 1, cols < N, other=0).to(tl.float32) * ps
        c2 = tl.load(P + cols * 6 + 2, cols < N, other=0).to(tl.float32) * ps
        c3 = tl.load(P + cols * 6 + 3, cols < N, other=0).to(tl.float32) * ps
        c4 = tl.load(P + cols * 6 + 4, cols < N, other=0).to(tl.float32) * ps
        c5 = tl.load(P + cols * 6 + 5, cols < N, other=0).to(tl.float32) * ps
        if RANK == 1:
            mean = x
        else:
            mean = tl.load(MEAN+rows[:,None]*(N//RANK)+cols[None,:]//RANK,mask,other=0.)
        t = 4. * libdevice.tanh(mean * .25)
        z = libdevice.tanh(scale[None, :] * (t + shift[None, :]))
        cubic = 2*c0 - 2*c1 + c2 + c3
        quadratic = -3*c0 + 3*c1 - 2*c2 - c3
        spine = ((cubic*t + quadratic)*t + c2)*t + c0
        warped = ((4*c5*z + 2*c4)*z - 3*c5)*z - c4
        # Smooth FP32 surrogate for FP16 rounding and the INT8 lookup path.
        dpre = value * 2. * tl.maximum(x + spine + warped, 0.)
        dpre = tl.where(mask, dpre, 0.)
        dpoly = (3*cubic*t + 2*quadratic)*t + c2
        dwarp = (12*c5*z + 4*c4)*z - 3*c5
        dz = dwarp * (1. - z*z)
        dmean = dpre*(dpoly+dz*scale)*(1.-t*t*.0625)
        if RANK == 1:
            value = tl.where((original>=-65504.) & (original<=65504.),dpre+dmean,0.)
        else:
            value = dpre
            tl.store(DMEAN+rows[:,None]*N+cols[None,:],dmean,mask)

        # One deterministic partial per row tile, chart and learned scalar.
        offset = (tl.program_id(0) * N + cols) * 8
        t2 = t*t
        t3 = t2*t
        tl.store(PARTIAL + offset, tl.sum(dpre*(2*t3-3*t2+1), 0), cols < N)
        tl.store(PARTIAL + offset+1, tl.sum(dpre*(-2*t3+3*t2), 0), cols < N)
        tl.store(PARTIAL + offset+2, tl.sum(dpre*(t3-2*t2+t), 0), cols < N)
        tl.store(PARTIAL + offset+3, tl.sum(dpre*(t3-t2), 0), cols < N)
        tl.store(PARTIAL + offset+4, tl.sum(dpre*(2*z*z-1), 0), cols < N)
        tl.store(PARTIAL + offset+5, tl.sum(dpre*(4*z*z*z-3*z), 0), cols < N)
        raw_scale = tl.load(RAW_SCALE + cols//RANK, cols < N, other=0)
        raw_shift = tl.load(RAW_SHIFT + cols//RANK, cols < N, other=0)
        dscale = dpre*dz*(t+shift)*tl.sigmoid(raw_scale)
        shift_tanh = libdevice.tanh(raw_shift)
        dshift = dpre*dz*scale*.5*(1.-shift_tanh*shift_tanh)
        tl.store(PARTIAL + offset+6, tl.sum(dscale, 0), cols < N)
        tl.store(PARTIAL + offset+7, tl.sum(dshift, 0), cols < N)
    tl.store(OUT + rows[:, None] * N + cols[None, :], value,
             (rows[:, None] < M) & (cols[None, :] < N))


@tr.jit
def _reduce_geometry(PARTIAL, DC, DS, DH, N: tl.constexpr, TILES: tl.constexpr,
                     BLOCK: tl.constexpr):
    chart = tl.program_id(0)
    rows = tl.arange(0, BLOCK)
    component = tl.arange(0, 8)
    values = tl.load(PARTIAL + (rows[:, None]*N+chart)*8+component[None, :],
                     rows[:, None] < TILES, other=0)
    total = tl.sum(values, 0)
    tl.store(DC + chart*6+component, total, component < 6)
    tl.store(DS + chart, tl.sum(tl.where(component == 6, total, 0.), 0))
    tl.store(DH + chart, tl.sum(tl.where(component == 7, total, 0.), 0))


@tr.jit
def _finish_backward(G,DC,QX,SX,QD,SD,H,HS,PARTIAL,DX,DD,DU,DP,DS,DH,
                     M:tl.constexpr,I:tl.constexpr,O:tl.constexpr,N:tl.constexpr,
                     TILES:tl.constexpr,BLOCK:tl.constexpr):
    """Run three independent VJP GEMMs and geometry reduction in one launch."""
    pid=tl.program_id(0)
    tx=tl.cdiv(M,16)*tl.cdiv(I,32)
    td=tl.cdiv(N,16)*tl.cdiv(I,32)
    tu=tl.cdiv(O,16)*tl.cdiv(N,32)
    if pid<tx:
        a=pid//tl.cdiv(I,32);b=pid%tl.cdiv(I,32)
        value=_gradient_dot(DC,QD,SD,a,b,M,I,N,N,1,32,16,32,32)
        rows=a*16+tl.arange(0,16);cols=b*32+tl.arange(0,32)
        tl.store(DX+rows[:,None]*I+cols[None,:],value,(rows[:,None]<M)&(cols[None,:]<I))
    elif pid<tx+td:
        idx=pid-tx;a=idx//tl.cdiv(I,32);b=idx%tl.cdiv(I,32)
        value=_gradient_dot(DC,QX,SX,a,b,N,I,M,1,N,32,16,32,32)
        rows=a*16+tl.arange(0,16);cols=b*32+tl.arange(0,32)
        tl.store(DD+rows[:,None]*I+cols[None,:],value,(rows[:,None]<N)&(cols[None,:]<I))
    elif pid<tx+td+tu:
        idx=pid-tx-td;a=idx//tl.cdiv(N,32);b=idx%tl.cdiv(N,32)
        value=_gradient_dot(G,H,HS,a,b,O,N,M,1,O,32,16,32,32)
        rows=a*16+tl.arange(0,16);cols=b*32+tl.arange(0,32)
        tl.store(DU+rows[:,None]*N+cols[None,:],value,(rows[:,None]<O)&(cols[None,:]<N))
    else:
        chart=pid-tx-td-tu
        reduce_rows=tl.arange(0,BLOCK);component=tl.arange(0,8)
        values=tl.load(PARTIAL+(reduce_rows[:,None]*N+chart)*8+component[None,:],reduce_rows[:,None]<TILES,other=0.)
        total=tl.sum(values,0)
        tl.store(DP+chart*6+component,total,component<6)
        tl.store(DS+chart,tl.sum(tl.where(component==6,total,0.),0))
        tl.store(DH+chart,tl.sum(tl.where(component==7,total,0.),0))


def _gradient_mm(a, packed_b, scales, group):
    """Multiply an FP32 gradient by an INT8 saved operand without unpacking it."""
    out = torch.empty((a.shape[0], packed_b.shape[1]), device=a.device, dtype=torch.float32)
    _backward_mm[(tr.cdiv(out.shape[0], 16), tr.cdiv(out.shape[1], 32))](
        a, packed_b, scales, out, out, out, out, out, out, out, out, out, out, out, 1,
        out.shape[0], out.shape[1], a.shape[1], *a.stride(), group, False,
        BM=16,BN=32,BK=32,num_warps=4)
    return out


@tr.jit
def _finish_coordinate_gradient(C, DC, DM, M:tl.constexpr, N:tl.constexpr,
                                RANK:tl.constexpr, BLOCK:tl.constexpr=256):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=i//N;col=i%N
    x=tl.load(C+i,i<M*N,other=0.)
    direct=tl.load(DC+i,i<M*N,other=0.)
    shared=tl.load(DM+row*(N//RANK)+col//RANK,i<M*N,other=0.)
    result=tl.where((x>=-65504.)&(x<=65504.),direct+shared,0.)
    tl.store(DC+i,result,i<M*N)


class _KS3Train(torch.autograd.Function):
    """GS32 W8A8 forward and fused first-order FP32 surrogate VJP."""
    @staticmethod
    def forward(ctx,inputs,down,up,coefficients,raw_scale,raw_shift,
                lut,nonlinear,world,reducer,subspace_rank):
        tensors=(down,up,coefficients,raw_scale,raw_shift)
        if any(t.device!=inputs.device or t.dtype!=torch.float32 for t in tensors):
            raise ValueError('KS3 training requires FP32 master parameters on the input device.')
        if inputs.dtype not in (torch.float32,torch.float16,torch.bfloat16):
            raise ValueError('KS3 inputs must be FP32, FP16 or BF16.')
        ctx.input_dtype=inputs.dtype
        output_dtype=torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else inputs.dtype
        if not inputs.is_cuda or inputs.shape[-1]!=down.shape[1]:
            raise ValueError('KS3 training requires GPU inputs with the configured feature width.')
        x=inputs.reshape(-1,inputs.shape[-1]).contiguous()
        raw_scale,raw_shift=raw_scale.contiguous(),raw_shift.contiguous()
        m,local=x.shape[0],down.shape[0]
        ctx.input_shape,ctx.world,ctx.reducer=inputs.shape,world,reducer
        ctx.subspace_rank,ctx.coefficient_shape=subspace_rank,coefficients.shape
        ctx.empty=not m or not local
        if ctx.empty:
            ctx.save_for_backward(inputs,down,up,coefficients,raw_scale,raw_shift)
            y=torch.zeros((m,up.shape[0]),device=x.device,dtype=torch.float32)
        else:
            qx,sx,qd,sd,qu,su,p,ps,scale,shift=_pack_training_tensors(
                x,down,up,coefficients,raw_scale,raw_shift,subspace_rank)
            h,hs,coords,mean=_project_features(qx,sx,qd,sd,p,ps,scale,shift,lut,
                                              subspace_rank,nonlinear,save=True)
            y=torch.empty((m,up.shape[0]),device=x.device,dtype=torch.float32)
            _gemm[(tr.cdiv(m,16),tr.cdiv(up.shape[0],32))](
                h,hs,qu,su,y,y,y,y,y,y,y,
                m,up.shape[0],local,32,16,32,32,False,0,0,COORD=y,SAVE_COORD=False,num_warps=4)
            ctx.save_for_backward(qx,sx,qd,sd,qu,su,h,hs,coords,p,ps,scale,shift,
                                  raw_scale,raw_shift,mean)
        if world>1 and y.numel():
            if reducer is None:dist.all_reduce(y)
            else:y=reducer(y)
        return y.reshape(*inputs.shape[:-1],up.shape[0]).to(output_dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        rank=ctx.subspace_rank
        if ctx.empty:
            grads=[torch.zeros_like(t) for t in ctx.saved_tensors]
            dx,ddown,dup,dcoeff,dscale,dshift=grads
        else:
            (qx,sx,qd,sd,qu,su,h,hs,coords,p,ps,scale,shift,
             raw_scale,raw_shift,mean)=ctx.saved_tensors
            grad=grad.reshape(-1,qu.shape[0]).contiguous()
            m,local=coords.shape
            dc=torch.empty_like(coords)
            dmean=torch.empty_like(coords) if rank>1 else dc
            partial=torch.empty((tr.cdiv(m,16),local,8),device=grad.device,dtype=torch.float32)
            _backward_mm[(tr.cdiv(m,16),tr.cdiv(local,32))](
                grad,qu,su,dc,coords,p,ps,scale,shift,raw_scale,raw_shift,partial,mean,dmean,rank,
                m,local,qu.shape[0],*grad.stride(),32,True,BM=16,BN=32,BK=32,num_warps=4)
            dcoeff=torch.empty((local,6),device=grad.device,dtype=torch.float32)
            dscale=torch.empty((local,),device=grad.device,dtype=torch.float32)
            dshift=torch.empty_like(dscale)
            if rank>1:
                shared=dmean.reshape(m,local//rank,rank).mean(-1)
                _finish_coordinate_gradient[(tr.cdiv(m*local,256),)](coords,dc,shared,m,local,rank,BLOCK=256)
            i,o=qd.shape[1],qu.shape[0]
            dx=torch.empty((m,i),device=grad.device,dtype=torch.float32)
            ddown=torch.empty((local,i),device=grad.device,dtype=torch.float32)
            dup=torch.empty((o,local),device=grad.device,dtype=torch.float32)
            tiles=(tr.cdiv(m,16)*tr.cdiv(i,32)+tr.cdiv(local,16)*tr.cdiv(i,32)
                   +tr.cdiv(o,16)*tr.cdiv(local,32)+local)
            _finish_backward[(tiles,)](grad,dc,qx,sx,qd,sd,h,hs,partial,dx,ddown,dup,dcoeff,dscale,dshift,
                m,i,o,local,partial.shape[0],tr.next_power_of_2(partial.shape[0]),num_warps=4)
            if rank>1:
                dscale=dscale.reshape(-1,rank).sum(-1)
                dshift=dshift.reshape(-1,rank).sum(-1)
            dcoeff=dcoeff.reshape(ctx.coefficient_shape)
            dx=dx.reshape(ctx.input_shape)
        if ctx.world>1 and dx.numel():
            if ctx.reducer is None:dist.all_reduce(dx)
            else:dx=ctx.reducer(dx)
        return dx.to(ctx.input_dtype),ddown,dup,dcoeff,dscale,dshift,None,None,None,None,None


def ks3_int8(inputs, down, up, coefficients, raw_scale, raw_shift, *,
             subspace_rank=1, nonlinear='fp16', tanh_lut=None):
    """Functional fused FFN for optimizer-owned parameters or parameter-bank views.

    Projection layouts are [charts*rank, input_dim] and [output_dim, charts*rank].
    Coefficients have six elements per projected coordinate; warp vectors have
    one per chart. Parameters remain FP32, while inputs may be FP32/FP16/BF16.
    Output follows autocast's dtype, or the input dtype outside autocast.
    No parameters are copied or registered and no distributed reduction occurs.
    """
    modes={'fp32':0,'fp16':1,'int8':2}
    if nonlinear not in modes:
        raise ValueError('nonlinear must be fp32, fp16 or int8')
    charts=raw_scale.numel()
    if (subspace_rank<1 or raw_scale.ndim!=1 or raw_shift.ndim!=1
            or raw_shift.numel()!=charts or down.shape[0]!=charts*subspace_rank):
        raise ValueError('Inconsistent chart count, rank or warp vectors')
    if (down.ndim!=2 or up.ndim!=2 or up.shape[1]!=down.shape[0]
            or coefficients.numel()!=6*down.shape[0]):
        raise ValueError('Inconsistent projection or coefficient shapes')
    if tanh_lut is None:
        tanh_lut=(torch.tanh(torch.arange(-128,128,device=inputs.device).float()*(3.2/127.))*127).round().clamp(-127,127).to(torch.int8)
    return _KS3Train.apply(inputs,down,up,coefficients,raw_scale,raw_shift,
                          tanh_lut,modes[nonlinear],1,None,subspace_rank)


def _fused_cell_forward(self, inputs):
    return ks3_int8(inputs,self.down.weight,self.up.weight,self.coefficients,
                    self.raw_scale,self.raw_shift,subspace_rank=self.subspace_rank,
                    nonlinear=self._ks3_nonlinear,tanh_lut=self._ks3_tanh_lut)


def fuse_training(cell, nonlinear='fp16'):
    """Enable fused W8A8 training on a Torch KS3 cell without replacing parameters.

    Keeps down.weight/up.weight optimizer groups and the Torch state_dict format.
    Call once before torch.compile/DDP. Saved weights can be loaded into a fresh
    KS3 cell, then fused again. The derived lookup table is not checkpointed.
    """
    from types import MethodType
    if nonlinear not in ('fp32','fp16','int8'):
        raise ValueError('nonlinear must be fp32, fp16 or int8')
    if not all(hasattr(cell,name) for name in ('down','up','coefficients','raw_scale','raw_shift','subspace_rank')):
        raise TypeError('fuse_training expects a Torch KS3 cell')
    if any(p.dtype!=torch.float32 for p in cell.parameters()):
        raise ValueError('Keep KS3 parameters FP32; use autocast for mixed precision.')
    lut=(torch.tanh(torch.arange(-128,128,device=cell.down.weight.device).float()*(3.2/127.))*127).round().clamp(-127,127).to(torch.int8)
    cell.register_buffer('_ks3_tanh_lut',lut,persistent=False)
    cell._ks3_nonlinear=nonlinear
    cell.forward=MethodType(_fused_cell_forward,cell)
    return cell


def capture_training(cell, sample_input, *, warmup=3):
    """Capture one fixed-shape forward/backward lane without changing parameters.

    Opt-in for training loops that use each cell once before its backward (as
    nanoGPT does). Multiple outstanding forwards on the captured lane are not
    supported. Capture after device/dtype placement and before compilation or
    DDP. Evaluation, autocast, and other input shapes use the ordinary fused
    path; ordinary fuse_training() has no outstanding-forward restriction.
    Checkpoints still contain only the normal KS3 parameters.
    """
    from types import MethodType
    if not hasattr(cell,'_ks3_nonlinear') or not cell.training:
        raise ValueError('Capture a fused KS3 cell in training mode')
    if not sample_input.is_cuda or not sample_input.requires_grad:
        raise ValueError('Capture needs a GPU sample with requires_grad=True')
    eager=cell.forward
    signature=(sample_input.shape,sample_input.dtype,sample_input.device)
    torch.cuda.make_graphed_callables(cell,(sample_input,),num_warmup_iters=warmup)
    replay=cell.forward
    def forward(self,inputs):
        if (self.training and torch.is_grad_enabled() and inputs.requires_grad
                and not torch.is_autocast_enabled('cuda')
                and (inputs.shape,inputs.dtype,inputs.device)==signature):
            return replay(inputs)
        return eager(inputs)
    cell.forward=MethodType(forward,cell)
    return cell


@tr.jit
def _adam_tensor(P,G,M,V,tile,N:tl.constexpr,B1:tl.constexpr,B2:tl.constexpr,
                  RATE,CORRECTION,DECAY,BLOCK:tl.constexpr):
    i=tile*BLOCK+tl.arange(0,BLOCK)
    p=tl.load(P+i,i<N,other=0.)
    g=tl.load(G+i,i<N,other=0.)
    m=tl.load(M+i,i<N,other=0.);v=tl.load(V+i,i<N,other=0.)
    # Match torch.lerp_ and add_(alpha=...) rounding explicitly; unrelated
    # expressions must not contract across the original optimizer's stages.
    m=tl.fma(1.-B1,g-m,m)
    v=tl.fma(1.-B2,g*g-v,v)
    update=tl.div_rn(m,libdevice.sqrt(v)+1e-10)*CORRECTION
    p=tl.fma(DECAY,tl.where(p*update>0.,p,0.),p)
    p=tl.fma(-RATE,update,p)
    tl.store(P+i,p,i<N);tl.store(M+i,m,i<N);tl.store(V+i,v,i<N)


@tr.jit
def _adam_geometry(P0,P1,P2,G0,G1,G2,M0,M1,M2,V0,V1,V2,
                    N0:tl.constexpr,N1:tl.constexpr,N2:tl.constexpr,
                    B1:tl.constexpr,B2:tl.constexpr,RATE,CORRECTION,DECAY,
                    BLOCK:tl.constexpr):
    pid=tl.program_id(0);t0=tl.cdiv(N0,BLOCK);t1=tl.cdiv(N1,BLOCK)
    if pid<t0:
        _adam_tensor(P0,G0,M0,V0,pid,N0,B1,B2,RATE,CORRECTION,DECAY,BLOCK)
    elif pid<t0+t1:
        _adam_tensor(P1,G1,M1,V1,pid-t0,N1,B1,B2,RATE,CORRECTION,DECAY,BLOCK)
    else:
        _adam_tensor(P2,G2,M2,V2,pid-t0-t1,N2,B1,B2,RATE,CORRECTION,DECAY,BLOCK)


@torch.no_grad()
def geometry_adam_step(groups, step, learning_rate):
    """Fuse the nanoGPT control optimizer's three KS3 geometry Adam updates.

    Uses its existing FP32 parameters, moments, bias correction, cautious
    decay and learning rates. Returns False when the generic optimizer must
    handle a missing gradient or unsupported layout/hyperparameter grouping.
    Projection ANVIL updates are deliberately outside this helper.
    """
    if len(groups)!=3:
        return False
    first=groups[0]
    if any(g['p'].grad is None or not g['p'].is_cuda
           or (g['beta'],g['mul'],g['wd'])!=(first['beta'],first['mul'],first['wd'])
           or any(t.dtype!=torch.float32 or not t.is_contiguous()
                  for t in (g['p'],g['p'].grad,g['m'],g['v'])) for g in groups):
        return False
    beta1,beta2=first['beta'];rate=learning_rate*first['mul']
    correction=math.sqrt(1-beta2**step)/(1-beta1**step)
    p=[g['p'] for g in groups];grad=[g['p'].grad for g in groups]
    m=[g['m'] for g in groups];v=[g['v'] for g in groups]
    sizes=[t.numel() for t in p]
    _adam_geometry[(sum(tr.cdiv(n,256) for n in sizes),)](
        *p,*grad,*m,*v,*sizes,beta1,beta2,rate,correction,-rate*rate*first['wd'],256,
        num_warps=4,enable_fp_fusion=False)
    return True


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
