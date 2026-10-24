"""Fused FP32 rank-8 stabilized KS2 geometry and analytic reverse mode.

The coordinate residual and input/output projections remain in their callers.
Gradient equations follow the existing independently tested CPU implementation
in experiments/common/ks2_cpu/backward.cpp. Partial parameter gradients are
reduced deterministically across token tiles, with no floating-point atomics.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice
from torch.autograd.function import once_differentiable

NAMES=('geo_spine','geo_t_basis_edge','geo_x_basis_edge','geo_den_delta','geo_den_weight',
       'geo_raw_scale','geo_raw_shift','geo_edge_raw_scale','geo_edge_raw_shift')
SIZES=(48,384,256,64,64,1,1,1,1)

@tr.jit
def _tanh(x):return libdevice.tanh(x)

@tr.jit
def _sp(x):return tl.where(x>20,x,tl.log(1+tl.exp(x)))

@tr.jit
def _basis(t,z):
    t2=t*t;t3=t2*t;z2=z*z
    return (2*t3-3*t2+1,-2*t3+3*t2,t3-2*t2+t,t3-t2,2*z2-1,4*z2*z-3*z)

@tr.jit
def _geometry_fwd(X,P,Y,N:tl.constexpr,S:tl.constexpr,B:tl.constexpr):
    block=tl.program_id(0);s=tl.program_id(1)
    n=block*B+tl.arange(0,B);j=tl.arange(0,8);e=tl.arange(0,64)
    x=tl.load(X+(n[:,None]*S+s)*8+j[None,:],n[:,None]<N,other=0)
    x=tl.minimum(tl.maximum(x,-65504.),65504.)
    mean=tl.sum(x,1)*.125;energy=tl.sum(x*x,1)
    p=P+s*820
    scale=_sp(tl.load(p+816))+1e-4;shift=.5*_tanh(tl.load(p+817))
    escale=_sp(tl.load(p+818))+1e-4;eshift=.5*_tanh(tl.load(p+819))
    t0=_tanh(mean*.25);t=4*t0;z=_tanh(scale*(t+shift));basis=_basis(t,z)
    spine=tl.full((B,8),0.,tl.float32)
    a=tl.full((B,64),0.,tl.float32)
    for k in tl.static_range(6):
        spine+=basis[k][:,None]*tl.load(p+k*8+j)[None,:]
        a+=basis[k][:,None]*tl.load(p+48+k*64+e)[None,:]
    xi=tl.broadcast_to(x[:,:,None],(B,8,8)).reshape(B,64)
    w=_tanh(escale*(xi+eshift));w2=w*w
    xb=(tl.full((B,64),1.,tl.float32),w,2*w2-1,4*w2*w-3*w)
    b=tl.full((B,64),0.,tl.float32)
    for k in tl.static_range(4):b+=xb[k]*tl.load(p+432+k*64+e)[None,:]
    delta=xi-mean[:,None];ds=delta*delta
    distance=ds/tl.maximum(ds+energy[:,None]*.125,1e-30)
    importance=xi*xi/tl.maximum(energy[:,None],1e-30)
    da=8*tl.sigmoid(tl.load(p+688+e)-2.086258912947793)
    db=8*tl.sigmoid(tl.load(p+752+e)-2.086258912947793)
    inv=1/(1+da[None,:]*distance+db[None,:]*importance)
    out=spine+tl.sum((a*b*inv).reshape(B,8,8),axis=1)
    tl.store(Y+(n[:,None]*S+s)*8+j[None,:],out,n[:,None]<N)

@tr.jit
def _geometry_bwd(X,P,DY,DX,DP,N:tl.constexpr,S:tl.constexpr,B:tl.constexpr):
    block=tl.program_id(0);s=tl.program_id(1)
    n=block*B+tl.arange(0,B);j=tl.arange(0,8);e=tl.arange(0,64)
    valid=n<N;offs=(n[:,None]*S+s)*8+j[None,:]
    raw=tl.load(X+offs,valid[:,None],other=0)
    x=tl.minimum(tl.maximum(raw,-65504.),65504.)
    u=tl.load(DY+offs,valid[:,None],other=0)
    mean=tl.sum(x,1)*.125;energy=tl.sum(x*x,1)
    p=P+s*820;dp=DP+(block*S+s)*820
    rs=tl.load(p+816);rh=tl.load(p+817);res=tl.load(p+818);reh=tl.load(p+819)
    scale=_sp(rs)+1e-4;rawshift=_tanh(rh);shift=.5*rawshift
    escale=_sp(res)+1e-4;raweshift=_tanh(reh);eshift=.5*raweshift
    t0=_tanh(mean*.25);t=4*t0;z=_tanh(scale*(t+shift));basis=_basis(t,z)
    xi=tl.broadcast_to(x[:,:,None],(B,8,8)).reshape(B,64);uj=tl.broadcast_to(u[:,None,:],(B,8,8)).reshape(B,64)
    w=_tanh(escale*(xi+eshift));w2=w*w
    xb=(tl.full((B,64),1.,tl.float32),w,2*w2-1,4*w2*w-3*w)
    dxb=(tl.full((B,64),0.,tl.float32),tl.full((B,64),1.,tl.float32),4*w,12*w2-3)
    a=tl.full((B,64),0.,tl.float32);b=tl.full((B,64),0.,tl.float32)
    for k in tl.static_range(6):a+=basis[k][:,None]*tl.load(p+48+k*64+e)[None,:]
    for k in tl.static_range(4):b+=xb[k]*tl.load(p+432+k*64+e)[None,:]
    delta=xi-mean[:,None];ds=delta*delta;rd=ds+energy[:,None]*.125
    ed=tl.maximum(energy[:,None],1e-30);dd=tl.maximum(rd,1e-30)
    distance=ds/dd;importance=xi*xi/ed
    da=8*tl.sigmoid(tl.load(p+688+e)-2.086258912947793)
    db=8*tl.sigmoid(tl.load(p+752+e)-2.086258912947793)
    inv=1/(1+da[None,:]*distance+db[None,:]*importance)
    ga=uj*b*inv;gb=uj*a*inv;gd=-(uj*a*inv)*(b*inv)
    tl.store(dp+688+e,tl.sum(gd*distance,0)*da*(1-da*.125))
    tl.store(dp+752+e,tl.sum(gd*importance,0)*db*(1-db*.125))
    dbasis0=tl.full((B,),0.,tl.float32);dbasis1=dbasis0;dbasis2=dbasis0
    dbasis3=dbasis0;dbasis4=dbasis0;dbasis5=dbasis0
    for k in tl.static_range(6):
        tl.store(dp+k*8+j,tl.sum(u*basis[k][:,None],0))
        tl.store(dp+48+k*64+e,tl.sum(ga*basis[k][:,None],0))
        d=tl.sum(u*tl.load(p+k*8+j)[None,:],1)+tl.sum(ga*tl.load(p+48+k*64+e)[None,:],1)
        if k==0:dbasis0=d
        elif k==1:dbasis1=d
        elif k==2:dbasis2=d
        elif k==3:dbasis3=d
        elif k==4:dbasis4=d
        else:dbasis5=d
    gw=tl.full((B,64),0.,tl.float32)
    for k in tl.static_range(4):
        tl.store(dp+432+k*64+e,tl.sum(gb*xb[k],0))
        gw+=gb*tl.load(p+432+k*64+e)[None,:]*dxb[k]
    gu=gw*(1-w2)
    tl.store(dp+818,tl.sum(tl.sum(gu*(xi+eshift),1),0)*tl.sigmoid(res))
    tl.store(dp+819,tl.sum(tl.sum(gu*escale,1),0)*.5*(1-raweshift*raweshift))
    dx=gu*escale
    distgrad=gd*da[None,:];impgrad=gd*db[None,:]
    denomgrad=tl.where(rd>=1e-30,distgrad*distance/dd,0.)
    ddelta=(distgrad/dd-denomgrad)*(2*delta)
    dx+=ddelta+impgrad*(2*xi/ed)
    dm=-tl.sum(ddelta,1)
    de=-tl.sum(denomgrad*.125+tl.where(energy[:,None]>=1e-30,impgrad*importance/ed,0.),1)
    gz=dbasis4*4*z+dbasis5*(12*z*z-3)
    gus=gz*(1-z*z)
    tl.store(dp+816,tl.sum(gus*(t+shift),0)*tl.sigmoid(rs))
    tl.store(dp+817,tl.sum(gus*scale,0)*.5*(1-rawshift*rawshift))
    t2=t*t
    gt=gus*scale+dbasis0*(6*t2-6*t)+dbasis1*(-6*t2+6*t)+dbasis2*(3*t2-4*t+1)+dbasis3*(3*t2-2*t)
    dm+=gt*(1-t0*t0)
    dx=tl.sum(dx.reshape(B,8,8),2)+dm[:,None]*.125+de[:,None]*(2*x)
    dx=tl.where((raw>=-65504.)&(raw<=65504.),dx,0.)
    tl.store(DX+offs,dx,valid[:,None])


class Geometry(torch.autograd.Function):
    @staticmethod
    def forward(ctx,coords,*parameters):
        assert coords.is_cuda and coords.dtype==torch.float32 and coords.shape[-1]==8
        s=coords.shape[-2];x=coords.contiguous();n=x.numel()//(s*8)
        packed=torch.cat([p.reshape(s,-1) for p in parameters],1)
        y=torch.empty_like(x)
        _geometry_fwd[(tr.cdiv(n,16),s)](x,packed,y,n,s,16,num_warps=4,enable_fp_fusion=False)
        ctx.save_for_backward(x,packed,*parameters)
        return y
    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        x,packed,*parameters=ctx.saved_tensors;s=x.shape[-2];n=x.numel()//(s*8)
        grad=grad.contiguous();dx=torch.empty_like(x)
        partial=torch.empty((tr.cdiv(n,16),s,820),device=x.device,dtype=x.dtype)
        _geometry_bwd[(tr.cdiv(n,16),s)](x,packed,grad,dx,partial,n,s,16,num_warps=4,enable_fp_fusion=False)
        dp=partial.sum(0)
        return (dx,*(g.reshape_as(p) for g,p in zip(dp.split(SIZES,1),parameters)))


def fused_geo_step(self,coords):return Geometry.apply(coords,*(getattr(self,name) for name in NAMES))


def install(cell):
    from types import MethodType
    assert cell.subspace_rank==8 and cell.geo_steps==1
    cell.geo_step=MethodType(fused_geo_step,cell)
    return cell
