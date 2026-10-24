"""Fused FP32 KS2 geometry for power-of-two ranks and equivalent cubic bases.

Derived from the separately validated rank8 analytic reverse-mode kernels.
Each token tile has approximately128coordinates, keeping small ranks occupied.
No atomics are used; parameter gradients reduce deterministic token tiles.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice
from torch.autograd.function import once_differentiable

NAMES=('geo_spine','geo_t_basis_edge','geo_x_basis_edge','geo_den_delta','geo_den_weight',
       'geo_raw_scale','geo_raw_shift','geo_edge_raw_scale','geo_edge_raw_shift')

@tr.jit
def _tanh(x):return libdevice.tanh(x)

@tr.jit
def _sp(x):return tl.where(x>20,x,tl.log(1+tl.exp(x)))

@tr.jit
def _basis(t,z,KIND:tl.constexpr):
    t2=t*t;t3=t2*t;z2=z*z
    if KIND==1:return (tl.full(t.shape,1.,tl.float32),t,t2,t3,2*z2-1,4*z2*z-3*z)
    elif KIND==2:return (tl.full(t.shape,1.,tl.float32),t,t2*.5,t3/6.,2*z2-1,4*z2*z-3*z)
    else:return (2*t3-3*t2+1,-2*t3+3*t2,t3-2*t2+t,t3-t2,2*z2-1,4*z2*z-3*z)

@tr.jit
def _polynomial_gradient(t,d0,d1,d2,d3,KIND:tl.constexpr):
    if KIND==1:return d1+2*t*d2+3*t*t*d3
    elif KIND==2:return d1+t*d2+.5*t*t*d3
    else:return d0*(6*t*t-6*t)+d1*(-6*t*t+6*t)+d2*(3*t*t-4*t+1)+d3*(3*t*t-2*t)

@tr.jit
def _geometry_fwd(X,P,Y,N:tl.constexpr,S:tl.constexpr,B:tl.constexpr,R:tl.constexpr,KIND:tl.constexpr):
    block=tl.program_id(0);s=tl.program_id(1)
    n=block*B+tl.arange(0,B);j=tl.arange(0,R);e=tl.arange(0,R*R)
    x=tl.load(X+(n[:,None]*S+s)*R+j[None,:],n[:,None]<N,other=0)
    x=tl.minimum(tl.maximum(x,-65504.),65504.)
    mean=tl.sum(x,1)/R;energy=tl.sum(x*x,1)
    p=P+s*(6*R+12*R*R+4)
    scale=_sp(tl.load(p+(6*R+12*R*R)))+1e-4;shift=.5*_tanh(tl.load(p+(6*R+12*R*R+1)))
    escale=_sp(tl.load(p+(6*R+12*R*R+2)))+1e-4;eshift=.5*_tanh(tl.load(p+(6*R+12*R*R+3)))
    t0=_tanh(mean*.25);t=4*t0;z=_tanh(scale*(t+shift));basis=_basis(t,z,KIND)
    spine=tl.full((B,R),0.,tl.float32)
    a=tl.full((B,R*R),0.,tl.float32)
    for k in tl.static_range(6):
        spine+=basis[k][:,None]*tl.load(p+k*R+j)[None,:]
        a+=basis[k][:,None]*tl.load(p+(6*R)+k*R*R+e)[None,:]
    xi=tl.broadcast_to(x[:,:,None],(B,R,R)).reshape(B,R*R)
    w=_tanh(escale*(xi+eshift));w2=w*w
    xb=(tl.full((B,R*R),1.,tl.float32),w,2*w2-1,4*w2*w-3*w)
    b=tl.full((B,R*R),0.,tl.float32)
    for k in tl.static_range(4):b+=xb[k]*tl.load(p+(6*R+6*R*R)+k*R*R+e)[None,:]
    delta=xi-mean[:,None];ds=delta*delta
    distance=ds/tl.maximum(ds+energy[:,None]/R,1e-30)
    importance=xi*xi/tl.maximum(energy[:,None],1e-30)
    da=8*tl.sigmoid(tl.load(p+(6*R+10*R*R)+e)-2.086258912947793)
    db=8*tl.sigmoid(tl.load(p+(6*R+11*R*R)+e)-2.086258912947793)
    inv=1/(1+da[None,:]*distance+db[None,:]*importance)
    out=spine+tl.sum((a*b*inv).reshape(B,R,R),axis=1)
    tl.store(Y+(n[:,None]*S+s)*R+j[None,:],out,n[:,None]<N)

@tr.jit
def _geometry_bwd(X,P,DY,DX,DP,N:tl.constexpr,S:tl.constexpr,B:tl.constexpr,R:tl.constexpr,KIND:tl.constexpr,MODE:tl.constexpr):
    block=tl.program_id(0);s=tl.program_id(1)
    n=block*B+tl.arange(0,B);j=tl.arange(0,R);e=tl.arange(0,R*R)
    valid=n<N;offs=(n[:,None]*S+s)*R+j[None,:]
    raw=tl.load(X+offs,valid[:,None],other=0)
    x=tl.minimum(tl.maximum(raw,-65504.),65504.)
    u=tl.load(DY+offs,valid[:,None],other=0)
    mean=tl.sum(x,1)/R;energy=tl.sum(x*x,1)
    p=P+s*(6*R+12*R*R+4);dp=DP+(block*S+s)*((6*R+12*R*R+4) if MODE==0 else 4)
    rs=tl.load(p+(6*R+12*R*R));rh=tl.load(p+(6*R+12*R*R+1));res=tl.load(p+(6*R+12*R*R+2));reh=tl.load(p+(6*R+12*R*R+3))
    scale=_sp(rs)+1e-4;rawshift=_tanh(rh);shift=.5*rawshift
    escale=_sp(res)+1e-4;raweshift=_tanh(reh);eshift=.5*raweshift
    t0=_tanh(mean*.25);t=4*t0;z=_tanh(scale*(t+shift));basis=_basis(t,z,KIND)
    xi=tl.broadcast_to(x[:,:,None],(B,R,R)).reshape(B,R*R);uj=tl.broadcast_to(u[:,None,:],(B,R,R)).reshape(B,R*R)
    w=_tanh(escale*(xi+eshift));w2=w*w
    xb=(tl.full((B,R*R),1.,tl.float32),w,2*w2-1,4*w2*w-3*w)
    dxb=(tl.full((B,R*R),0.,tl.float32),tl.full((B,R*R),1.,tl.float32),4*w,12*w2-3)
    a=tl.full((B,R*R),0.,tl.float32);b=tl.full((B,R*R),0.,tl.float32)
    for k in tl.static_range(6):a+=basis[k][:,None]*tl.load(p+(6*R)+k*R*R+e)[None,:]
    for k in tl.static_range(4):b+=xb[k]*tl.load(p+(6*R+6*R*R)+k*R*R+e)[None,:]
    delta=xi-mean[:,None];ds=delta*delta;rd=ds+energy[:,None]/R
    ed=tl.maximum(energy[:,None],1e-30);dd=tl.maximum(rd,1e-30)
    distance=ds/dd;importance=xi*xi/ed
    da=8*tl.sigmoid(tl.load(p+(6*R+10*R*R)+e)-2.086258912947793)
    db=8*tl.sigmoid(tl.load(p+(6*R+11*R*R)+e)-2.086258912947793)
    inv=1/(1+da[None,:]*distance+db[None,:]*importance)
    ga=uj*b*inv;gb=uj*a*inv;gd=-(uj*a*inv)*(b*inv)
    if MODE==0:
        tl.store(dp+(6*R+10*R*R)+e,tl.sum(gd*distance,0)*da*(1-da*.125))
    if MODE==0:
        tl.store(dp+(6*R+11*R*R)+e,tl.sum(gd*importance,0)*db*(1-db*.125))
    dbasis0=tl.full((B,),0.,tl.float32);dbasis1=dbasis0;dbasis2=dbasis0
    dbasis3=dbasis0;dbasis4=dbasis0;dbasis5=dbasis0
    for k in tl.static_range(6):
        if MODE==0:
            tl.store(dp+k*R+j,tl.sum(u*basis[k][:,None],0))
        if MODE==0:
            tl.store(dp+(6*R)+k*R*R+e,tl.sum(ga*basis[k][:,None],0))
        d=tl.sum(u*tl.load(p+k*R+j)[None,:],1)+tl.sum(ga*tl.load(p+(6*R)+k*R*R+e)[None,:],1)
        if k==0:dbasis0=d
        elif k==1:dbasis1=d
        elif k==2:dbasis2=d
        elif k==3:dbasis3=d
        elif k==4:dbasis4=d
        else:dbasis5=d
    gw=tl.full((B,R*R),0.,tl.float32)
    for k in tl.static_range(4):
        if MODE==0:
            tl.store(dp+(6*R+6*R*R)+k*R*R+e,tl.sum(gb*xb[k],0))
        gw+=gb*tl.load(p+(6*R+6*R*R)+k*R*R+e)[None,:]*dxb[k]
    gu=gw*(1-w2)
    if MODE!=2:
        tl.store(dp+((6*R+12*R*R+2) if MODE==0 else 2),tl.sum(tl.sum(gu*(xi+eshift),1),0)*tl.sigmoid(res))
    if MODE!=2:
        tl.store(dp+((6*R+12*R*R+3) if MODE==0 else 3),tl.sum(tl.sum(gu*escale,1),0)*.5*(1-raweshift*raweshift))
    dx=gu*escale
    distgrad=gd*da[None,:];impgrad=gd*db[None,:]
    denomgrad=tl.where(rd>=1e-30,distgrad*distance/dd,0.)
    ddelta=(distgrad/dd-denomgrad)*(2*delta)
    dx+=ddelta+impgrad*(2*xi/ed)
    dm=-tl.sum(ddelta,1)
    de=-tl.sum(denomgrad/R+tl.where(energy[:,None]>=1e-30,impgrad*importance/ed,0.),1)
    gz=dbasis4*4*z+dbasis5*(12*z*z-3)
    gus=gz*(1-z*z)
    if MODE!=2:
        tl.store(dp+((6*R+12*R*R) if MODE==0 else 0),tl.sum(gus*(t+shift),0)*tl.sigmoid(rs))
    if MODE!=2:
        tl.store(dp+((6*R+12*R*R+1) if MODE==0 else 1),tl.sum(gus*scale,0)*.5*(1-rawshift*rawshift))
    t2=t*t
    gt=gus*scale+_polynomial_gradient(t,dbasis0,dbasis1,dbasis2,dbasis3,KIND)
    dm+=gt*(1-t0*t0)
    dx=tl.sum(dx.reshape(B,R,R),2)+dm[:,None]/R+de[:,None]*(2*x)
    dx=tl.where((raw>=-65504.)&(raw<=65504.),dx,0.)
    tl.store(DX+offs,dx,valid[:,None])


@tr.jit
def _spine_fwd(X,P,RS,RH,Y,N:tl.constexpr,S:tl.constexpr,B:tl.constexpr,R:tl.constexpr,KIND:tl.constexpr):
    block=tl.program_id(0);s=tl.program_id(1);n=block*B+tl.arange(0,B);j=tl.arange(0,R)
    offs=(n[:,None]*S+s)*R+j[None,:]
    x=tl.load(X+offs,n[:,None]<N,other=0);x=tl.minimum(tl.maximum(x,-65504.),65504.)
    mean=tl.sum(x,1)/R;t=4*_tanh(mean*.25)
    scale=_sp(tl.load(RS+s))+1e-4;shift=.5*_tanh(tl.load(RH+s));z=_tanh(scale*(t+shift));basis=_basis(t,z,KIND)
    y=tl.full((B,R),0.,tl.float32)
    for k in tl.static_range(6):y+=basis[k][:,None]*tl.load(P+s*(6*R)+k*R+j)[None,:]
    tl.store(Y+offs,y,n[:,None]<N)

@tr.jit
def _spine_bwd(X,P,RS,RH,DY,DX,DP,N:tl.constexpr,S:tl.constexpr,B:tl.constexpr,R:tl.constexpr,KIND:tl.constexpr,MODE:tl.constexpr):
    block=tl.program_id(0);s=tl.program_id(1);n=block*B+tl.arange(0,B);j=tl.arange(0,R)
    offs=(n[:,None]*S+s)*R+j[None,:];valid=n[:,None]<N
    raw=tl.load(X+offs,valid,other=0);x=tl.minimum(tl.maximum(raw,-65504.),65504.);u=tl.load(DY+offs,valid,other=0)
    mean=tl.sum(x,1)/R;t0=_tanh(mean*.25);t=4*t0
    rs=tl.load(RS+s);rh=_tanh(tl.load(RH+s));scale=_sp(rs)+1e-4;shift=.5*rh;z=_tanh(scale*(t+shift));basis=_basis(t,z,KIND)
    d0=tl.full((B,),0.,tl.float32);d1=d0;d2=d0;d3=d0;d4=d0;d5=d0
    dp=DP+(block*S+s)*((6*R+2) if MODE==0 else 2)
    for k in tl.static_range(6):
        if MODE==0:
            tl.store(dp+k*R+j,tl.sum(u*basis[k][:,None],0))
        d=tl.sum(u*tl.load(P+s*(6*R)+k*R+j)[None,:],1)
        if k==0:d0=d
        elif k==1:d1=d
        elif k==2:d2=d
        elif k==3:d3=d
        elif k==4:d4=d
        else:d5=d
    gz=d4*4*z+d5*(12*z*z-3);gu=gz*(1-z*z)
    if MODE!=2:
        tl.store(dp+((6*R) if MODE==0 else 0),tl.sum(gu*(t+shift),0)*tl.sigmoid(rs))
    if MODE!=2:
        tl.store(dp+((6*R+1) if MODE==0 else 1),tl.sum(gu*scale,0)*.5*(1-rh*rh))
    t2=t*t;gt=gu*scale+_polynomial_gradient(t,d0,d1,d2,d3,KIND)
    dx=gt[:,None]*(1-t0[:,None]*t0[:,None])/R
    dx=tl.where((raw>=-65504.)&(raw<=65504.),dx,0.)
    tl.store(DX+offs,dx,valid)



class Geometry(torch.autograd.Function):
    @staticmethod
    def forward(ctx,coords,kind,*parameters):
        assert coords.is_cuda and coords.dtype==torch.float32
        r=coords.shape[-1];assert r in (1,2,4,8,16)
        s=coords.shape[-2];x=coords.contiguous();n=x.numel()//(s*r);b=128//r
        packed=torch.cat([p.reshape(s,-1) for p in parameters],1);y=torch.empty_like(x)
        _geometry_fwd[(tr.cdiv(n,b),s)](x,packed,y,n,s,b,r,kind,num_warps=4,enable_fp_fusion=False)
        ctx.save_for_backward(x,packed,*parameters);ctx.kind=kind
        return y
    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        x,packed,*parameters=ctx.saved_tensors;s=x.shape[-2];r=x.shape[-1];n=x.numel()//(s*r);b=128//r
        sizes=(6*r,6*r*r,4*r*r,r*r,r*r,1,1,1,1)
        needed=ctx.needs_input_grad[2:]
        mode=2 if not any(needed) else (1 if not any(needed[:-4]) else 0)
        dx=torch.empty_like(x)
        shape=(1,) if mode==2 else (tr.cdiv(n,b),s,4 if mode==1 else sum(sizes))
        partial=torch.empty(shape,device=x.device,dtype=x.dtype)
        _geometry_bwd[(tr.cdiv(n,b),s)](x,packed,grad.contiguous(),dx,partial,n,s,b,r,ctx.kind,mode,num_warps=4,enable_fp_fusion=False)
        if mode==2:return (dx,None,*([None]*len(parameters)))
        dp=partial.sum(0)
        if mode==1:return (dx,None,*([None]*5),*(dp[:,i].reshape_as(p) for i,p in enumerate(parameters[-4:])))
        return (dx,None,*(g.reshape_as(p) for g,p in zip(dp.split(sizes,1),parameters)))


class SpineGeometry(torch.autograd.Function):
    @staticmethod
    def forward(ctx,coords,spine,scale,shift,kind):
        assert coords.is_cuda and coords.dtype==torch.float32
        x=coords.contiguous();s=x.shape[-2];r=x.shape[-1];assert r in (1,2,4,8,16)
        n=x.numel()//(s*r);b=128//r;y=torch.empty_like(x)
        _spine_fwd[(tr.cdiv(n,b),s)](x,spine,scale,shift,y,n,s,b,r,kind,num_warps=4,enable_fp_fusion=False)
        ctx.save_for_backward(x,spine,scale,shift);ctx.kind=kind
        return y
    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        x,spine,scale,shift=ctx.saved_tensors;s=x.shape[-2];r=x.shape[-1];n=x.numel()//(s*r);b=128//r
        needed=ctx.needs_input_grad[1:4]
        mode=2 if not any(needed) else (1 if not needed[0] else 0)
        dx=torch.empty_like(x)
        shape=(1,) if mode==2 else (tr.cdiv(n,b),s,2 if mode==1 else 6*r+2)
        partial=torch.empty(shape,device=x.device,dtype=x.dtype)
        _spine_bwd[(tr.cdiv(n,b),s)](x,spine,scale,shift,grad.contiguous(),dx,partial,n,s,b,r,ctx.kind,mode,num_warps=4,enable_fp_fusion=False)
        if mode==2:return dx,None,None,None,None
        dp=partial.sum(0)
        if mode==1:return dx,None,dp[:,0].reshape_as(scale),dp[:,1].reshape_as(shift),None
        return dx,dp[:,:6*r].reshape_as(spine),dp[:,6*r].reshape_as(scale),dp[:,6*r+1].reshape_as(shift),None


def fused_geo_step(self,coords):
    kind={'hermite':0,'powers':1,'taylor':2}[getattr(self,'study_basis','hermite')]
    if getattr(self,'terms',None)=='spine':return SpineGeometry.apply(coords,self.geo_spine,self.geo_raw_scale,self.geo_raw_shift,kind)
    return Geometry.apply(coords,kind,*(getattr(self,name) for name in NAMES))


def install(cell):
    from types import MethodType
    assert cell.subspace_rank in (1,2,4,8,16) and cell.geo_steps==1
    assert getattr(cell,'terms',None) in (None,'spine')
    cell.geo_step=MethodType(fused_geo_step,cell)
    return cell
