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
scales and outputs are FP32.

For training, pass ``trainable=True`` to KS3Int8, then create the optimizer
from that module's parameters. It owns FP32 master copies and repacks them
on every forward. Backward fuses the output-projection VJP with the geometry
VJP, reduces all eight geometry gradients, and computes both projection
gradients and the input gradient. Packed operands are dequantized inside
the backward GEMMs; backward products and accumulators are FP32, not INT8.

Quantization uses an identity straight-through estimator with detached scales.
All modes differentiate smooth FP32 KS3 geometry at the quantized coordinates
and coefficients. This is also the surrogate for FP16 rounding and INT8 LUTs;
it is not the literal derivative of discrete rounding. Training inputs and
masters must be FP32. First-order gradients only. ``to_inference()`` packs
the updated masters into an inference-only module without optimizer state.

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
from torch.autograd.function import once_differentiable
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
          BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,FUSED:tl.constexpr,NL:tl.constexpr,ACT:tl.constexpr,
          COORD=None,SAVE_COORD:tl.constexpr=False):
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
        if SAVE_COORD:
            tl.store(COORD+mi[:,None]*N+ni[None,:],result,(mi[:,None]<M)&valid[None,:])
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
    """Fused KS3 W8A8 inference or training, optionally sharded across GPUs.

    ``cell`` can be the standalone KS3 reference or the legacy rank-1 cell.
    Load its trained weights before constructing this module. The packed
    buffers support state_dict/save/load; conversion does not mutate the cell.
    With ``trainable=True``, optimize this module's FP32 ``master_*`` parameters.
    ``eval()`` keeps those masters current; use ``to_inference()`` to freeze them.

    TP assumes a replicated input and replicated loss/upstream gradient.
    Forward sums output partials; backward sums input-gradient partials only.
    Each rank owns its local parameter gradients. A custom reducer must return
    an FP32 sum tensor and support both forward and backward calls. Whole-model
    ``convert()`` remains an inference export; this training API covers the FFN.
    """
    def __init__(self,cell,rank=0,world=1,nonlinear='fp16',reducer=None,*,trainable=False):
        assert nonlinear in ('fp32','fp16','int8')
        super().__init__();assert cell.subspace_rank==1 and cell.num_subspaces==368
        assert world in (1,2,4) and 0<=rank<world;local=368//world;start=rank*local;end=start+local
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
        self.trainable = trainable
        if trainable:
            # FP32 optimizer-owned masters; forward repacks after every update.
            self.master_down = nn.Parameter(cell.down.weight[start:end].detach().float().clone())
            self.master_up = nn.Parameter(cell.up.weight[:,start:end].detach().float().clone())
            self.master_coefficients = nn.Parameter(coefficients.detach().reshape(368,6)[start:end].float().clone())
            self.master_raw_scale = nn.Parameter(raw_scale.detach().reshape(-1)[start:end].float().clone())
            self.master_raw_shift = nn.Parameter(raw_shift.detach().reshape(-1)[start:end].float().clone())
    def forward(self,x):
        if self.trainable:
            return _KS3Train.apply(
                x, self.master_down, self.master_up, self.master_coefficients,
                self.master_raw_scale, self.master_raw_shift, self.tanh_lut,
                self.nonlinear, self.world, self.reducer)
        q,s=quantize(x,128);m=x.numel()//x.shape[-1]
        h=torch.empty((*x.shape[:-1],self.local),device=x.device,dtype=torch.int8)
        hs=torch.empty((m,self.local//92),device=x.device)
        _gemm[(tr.cdiv(m,16),self.local//92)](q,s,self.down.qweight,self.down.scales,h,hs,self.spine,self.spine_scale,self.warp_scale,self.warp_shift,self.tanh_lut,m,self.local,x.shape[-1],128,16,128,128,True,self.nonlinear,0,num_warps=4)
        y=self.up.from_quantized(h,hs)
        if self.world>1:
            if self.reducer is None:dist.all_reduce(y)
            else:y=self.reducer(y)
        return y

    @torch.no_grad()
    def to_inference(self):
        """Return a frozen copy, packing the latest master weights on each rank."""
        result = copy.deepcopy(self)
        if not self.trainable:
            return result.eval()
        result.down = Int8Linear(self.master_down, 128)
        result.up = Int8Linear(self.master_up, 92)
        coeff = self.master_coefficients
        result.spine_scale = coeff.abs().amax(-1).clamp_min(1e-10)/127.
        result.spine = (coeff/result.spine_scale[:, None]).round().clamp(-127,127).to(torch.int8)
        result.warp_scale = torch.nn.functional.softplus(self.master_raw_scale)+1e-4
        result.warp_shift = .5*torch.tanh(self.master_raw_shift)
        for name in ('master_down', 'master_up', 'master_coefficients',
                     'master_raw_scale', 'master_raw_shift'):
            delattr(result, name)
        result.trainable = False
        return result.eval()


@tr.jit
def _backward_mm(A, B, BS, OUT, COORD, P, PS, SCALE, SHIFT, RAW_SCALE,
                 RAW_SHIFT, PARTIAL,
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
    inner = tl.arange(0, BK)
    value = tl.full((BM, BN), 0., tl.float32)
    for start in range(tr.cdiv(K, BK)):
        kk = start * BK + inner
        a = tl.load(A + rows[:, None] * A0 + kk[None, :] * A1,
                    (rows[:, None] < M) & (kk[None, :] < K), other=0)
        b = tl.load(B + kk[:, None] * N + cols[None, :],
                    (kk[:, None] < K) & (cols[None, :] < N), other=0)
        bs = tl.load(BS + kk[:, None] * tr.cdiv(N, GROUP) + cols[None, :] // GROUP,
                     (kk[:, None] < K) & (cols[None, :] < N), other=0)
        value += tl.dot(a.to(tl.float32), b.to(tl.float32) * bs,
                        input_precision="ieee")

    if GEOMETRY:
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        original = tl.load(COORD + rows[:, None] * N + cols[None, :], mask, other=0)
        x = tl.minimum(tl.maximum(original, -65504.), 65504.)
        scale = tl.load(SCALE + cols, cols < N, other=1)
        shift = tl.load(SHIFT + cols, cols < N, other=0)
        ps = tl.load(PS + cols, cols < N, other=0)
        c0 = tl.load(P + cols * 6, cols < N, other=0).to(tl.float32) * ps
        c1 = tl.load(P + cols * 6 + 1, cols < N, other=0).to(tl.float32) * ps
        c2 = tl.load(P + cols * 6 + 2, cols < N, other=0).to(tl.float32) * ps
        c3 = tl.load(P + cols * 6 + 3, cols < N, other=0).to(tl.float32) * ps
        c4 = tl.load(P + cols * 6 + 4, cols < N, other=0).to(tl.float32) * ps
        c5 = tl.load(P + cols * 6 + 5, cols < N, other=0).to(tl.float32) * ps
        t = 4. * libdevice.tanh(x * .25)
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
        dx = dpre * (1. + (dpoly + dz*scale) * (1. - t*t*.0625))
        dx = tl.where((original >= -65504.) & (original <= 65504.), dx, 0.)
        value = dx

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
        raw_scale = tl.load(RAW_SCALE + cols, cols < N, other=0)
        raw_shift = tl.load(RAW_SHIFT + cols, cols < N, other=0)
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


def _gradient_mm(a, packed_b, scales, group):
    """Multiply an FP32 gradient by an INT8 saved operand without unpacking it."""
    out = torch.empty((a.shape[0], packed_b.shape[1]), device=a.device, dtype=torch.float32)
    _backward_mm[(tr.cdiv(out.shape[0], 16), tr.cdiv(out.shape[1], 32))](
        a, packed_b, scales, out, out, out, out, out, out, out, out, out,
        out.shape[0], out.shape[1], a.shape[1], *a.stride(), group, False,
        num_warps=4)
    return out


class _KS3Train(torch.autograd.Function):
    """W8A8 forward, identity quantization STE, and fused first-order FP32 VJP."""

    @staticmethod
    def forward(ctx, inputs, down, up, coefficients, raw_scale, raw_shift,
                lut, nonlinear, world, reducer):
        tensors = (inputs, down, up, coefficients, raw_scale, raw_shift)
        if any(t.device != inputs.device or t.dtype != torch.float32 for t in tensors):
            raise ValueError('KS3 training requires FP32 inputs and master parameters on one device.')
        if not inputs.is_cuda or inputs.shape[-1] != down.shape[1]:
            raise ValueError('KS3 training requires GPU inputs with the configured feature width.')
        if not inputs.numel():
            raise ValueError('KS3 fused training requires at least one input token.')
        x = inputs.reshape(-1, inputs.shape[-1]).contiguous()
        m, local = x.shape[0], down.shape[0]
        qx, sx = quantize(x, 128)
        qd, sd = pack_weight(down, 128)
        qu, su = pack_weight(up, 92)
        ps = coefficients.abs().amax(-1).clamp_min(1e-10)/127.
        p = (coefficients/ps[:, None]).round().clamp(-127,127).to(torch.int8)
        scale = torch.nn.functional.softplus(raw_scale)+1e-4
        shift = .5*torch.tanh(raw_shift)
        h = torch.empty((m, local), device=x.device, dtype=torch.int8)
        hs = torch.empty((m, local//92), device=x.device, dtype=torch.float32)
        coords = torch.empty((m, local), device=x.device, dtype=torch.float32)
        _gemm[(tr.cdiv(m,16),local//92)](
            qx,sx,qd,sd,h,hs,p,ps,scale,shift,lut,
            m,local,x.shape[1],128,16,128,128,True,nonlinear,0,
            COORD=coords,SAVE_COORD=True,num_warps=4)
        y = torch.empty((m,up.shape[0]), device=x.device, dtype=torch.float32)
        _gemm[(tr.cdiv(m,16),tr.cdiv(up.shape[0],32))](
            h,hs,qu,su,y,y,y,y,y,y,y,
            m,up.shape[0],local,92,16,32,128,False,0,0,num_warps=4)
        if world > 1:
            if reducer is None:
                dist.all_reduce(y)
            else:
                y = reducer(y)
        ctx.save_for_backward(qx,sx,qd,sd,qu,su,h,hs,coords,p,ps,scale,shift,
                              raw_scale,raw_shift)
        ctx.input_shape, ctx.world, ctx.reducer = inputs.shape, world, reducer
        return y.reshape(*inputs.shape[:-1], up.shape[0])

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        (qx,sx,qd,sd,qu,su,h,hs,coords,p,ps,scale,shift,
         raw_scale,raw_shift) = ctx.saved_tensors
        grad = grad.reshape(-1,qu.shape[0]).contiguous()
        m, local = coords.shape
        dc = torch.empty_like(coords)
        partial = torch.empty((tr.cdiv(m,16),local,8), device=grad.device, dtype=torch.float32)
        _backward_mm[(tr.cdiv(m,16),tr.cdiv(local,32))](
            grad,qu,su,dc,coords,p,ps,scale,shift,raw_scale,raw_shift,partial,
            m,local,qu.shape[0],*grad.stride(),92,True,num_warps=4)
        dcoeff = torch.empty((local,6), device=grad.device, dtype=torch.float32)
        dscale, dshift = torch.empty_like(raw_scale), torch.empty_like(raw_shift)
        _reduce_geometry[(local,)](
            partial,dcoeff,dscale,dshift,local,partial.shape[0],
            tr.next_power_of_2(partial.shape[0]),num_warps=4)
        dup = _gradient_mm(grad.T,h,hs,92)
        ddown = _gradient_mm(dc.T,qx,sx,128)
        dx = _gradient_mm(dc,qd,sd,128)
        # Output gradients are replicated, so the forward sum has identity VJP.
        # Only the contributions to the replicated input must be summed here.
        if ctx.world > 1:
            if ctx.reducer is None:
                dist.all_reduce(dx)
            else:
                dx = ctx.reducer(dx)
        return dx.reshape(ctx.input_shape),ddown,dup,dcoeff,dscale,dshift,None,None,None,None


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
