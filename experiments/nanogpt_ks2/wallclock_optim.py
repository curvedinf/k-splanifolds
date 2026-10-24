"""Captured execution of the existing FP32 ANVIL/Adam optimizer equations.

No trainable parameters, moments or matrix iterations are removed. Host-computed
schedule scalars and fresh gradients feed one reusable device graph per update.
"""
import math

import torch
import triton as tr
import triton.language as tl

from experiments.common.ks3_int8 import libdevice
from experiments.nanogpt_ks2.optim import MAPS


@tr.jit
def _adam_group(P,G,M,V,C,N:tl.constexpr,INDEX:tl.constexpr,
                 B1:tl.constexpr,B2:tl.constexpr,BLOCK:tl.constexpr,SCALE:tl.constexpr):
    rate=tl.load(C+3+3*INDEX);decay=tl.load(C+4+3*INDEX)
    correction=tl.load(C+5+3*INDEX)
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    p=tl.load(P+i,i<N,other=0.);g=tl.load(G+i,i<N,other=0.)/SCALE
    m=tl.load(M+i,i<N,other=0.);v=tl.load(V+i,i<N,other=0.)
    # Torch lerp chooses the endpoint that avoids cancellation. In particular,
    # beta1=0 for value embeddings must copy g exactly, including zero rows.
    if B1>.5:m=tl.fma(1.-B1,g-m,m)
    else:m=tl.fma(-B1,g-m,g)
    v=tl.fma(1.-B2,g*g-v,v)
    update=tl.div_rn(m,libdevice.sqrt(v)+1e-10)*correction
    p=tl.fma(-decay,tl.where(p*update>0.,p,0.),p)
    p=tl.fma(-rate,update,p)
    tl.store(P+i,p,i<N);tl.store(M+i,m,i<N);tl.store(V+i,v,i<N)


class GraphOptimizer:
    def __init__(self, optimizer, *, batch_matrices=False, fuse_adam=False, gradient_scale=1.,scale_in_update=False):
        self.base=optimizer
        self.matrices,self.adam=optimizer.matrices,optimizer.adam
        self.groups=self.matrices+self.adam
        self.parameters=[s['p'] for s in self.groups]
        self.grad_buffers=[torch.zeros_like(p) for p in self.parameters]
        self.batch_matrices=batch_matrices;self.fuse_adam=fuse_adam
        self.gradient_scale=gradient_scale
        self.scale_in_update=scale_in_update
        self.update_scale=gradient_scale if scale_in_update else 1.
        device=self.parameters[0].device
        self.controls=torch.zeros(3+3*len(self.groups),device=device,dtype=torch.float32)
        self.controls[:3]=torch.tensor([.85,.85,1.],device=device)
        self.buckets=[]
        if batch_matrices:
            groups={}
            for i,s in enumerate(self.matrices):
                groups.setdefault((tuple(s['fast'].shape),s['axis']),[]).append(i)
            for (_,axis),indices in groups.items():
                bucket=dict(indices=indices,axis=axis,index=torch.tensor(indices,device=device))
                for name in ('fast','slow','energy'):
                    bucket[name]=torch.stack([self.matrices[i][name] for i in indices])
                    for row,i in enumerate(indices):self.matrices[i][name]=bucket[name][row]
                self.buckets.append(bucket)
        self.fallbacks=0
        # Zero gradients, moments and learning rates leave a fresh optimizer
        # and its model unchanged while kernels warm and the graph captures.
        if optimizer.t!=0:
            raise ValueError('Capture a fresh optimizer before its first update')
        stream=torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):self._update()
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self._update()
        torch.cuda.synchronize(device)

    def zero_grad(self):
        self.base.zero_grad()

    @torch.no_grad()
    def _update(self):
        beta,fast_beta,mix=self.controls[0],self.controls[1],self.controls[2]
        if self.batch_matrices:
            self._update_batched(beta,fast_beta,mix)
        for i,s in enumerate(self.matrices if not self.batch_matrices else []):
            p=s['p'].mT if s['transpose'] else s['p']
            g=self.grad_buffers[i]
            if self.update_scale!=1.:g=g/self.update_scale
            g=g.mT if s['transpose'] else g
            s['fast'].lerp_(g,1-fast_beta)
            s['slow'].lerp_(g,.02)
            x=g.lerp(mix*s['fast']+(1-mix)*s['slow'],beta)
            x=x/(1.05*x.norm(dim=(-2,-1),keepdim=True)+1e-6)
            tall=x.shape[-2]>x.shape[-1]
            for a,b,c in MAPS:
                gram=x.mT@x if tall else x@x.mT
                poly=b*gram+c*(gram@gram)
                x=a*x+(x@poly if tall else poly@x)
            power=x.square().mean(s['axis'],keepdim=True)
            s['energy'].lerp_(power,.1)
            gain=s['energy'].clamp_min(1e-10).rsqrt()
            ratio=(power.sum((-2,-1),keepdim=True)/(power*gain.square()).sum((-2,-1),keepdim=True).clamp_min(1e-10)).sqrt()
            update=x*gain*ratio
            rate,decay=self.controls[3+3*i],self.controls[4+3*i]
            p.addcmul_(p*(p*s['slow']>=0),decay,value=-1)
            p.addcmul_(update,rate,value=-1)
        for j,s in enumerate(self.adam):
            i=len(self.matrices)+j;p=s['p'];b=s['beta'];g=self.grad_buffers[i]
            if self.fuse_adam:
                _adam_group[(tr.cdiv(p.numel(),1024),)](p,g,s['m'],s['v'],self.controls,
                    p.numel(),i,b[0],b[1],1024,self.update_scale,num_warps=4,enable_fp_fusion=False)
                continue
            if self.update_scale!=1.:g=g/self.update_scale
            s['m'].lerp_(g,1-b[0]);s['v'].lerp_(g.square(),1-b[1])
            rate,decay,correction=self.controls[3+3*i],self.controls[4+3*i],self.controls[5+3*i]
            update=s['m']/(s['v'].sqrt()+1e-10)*correction
            p.addcmul_(p*(p*update>0),decay,value=-1)
            p.addcmul_(update,rate,value=-1)

    @torch.no_grad()
    def _update_batched(self,beta,fast_beta,mix):
        for bucket in self.buckets:
            indices=bucket['indices']
            params=[self.matrices[i]['p'].mT if self.matrices[i]['transpose'] else self.matrices[i]['p'] for i in indices]
            grads=[self.grad_buffers[i].mT if self.matrices[i]['transpose'] else self.grad_buffers[i] for i in indices]
            p=torch.stack(params);g=torch.stack(grads)
            if self.update_scale!=1.:g=g/self.update_scale
            fast,slow,energy=bucket['fast'],bucket['slow'],bucket['energy']
            fast.lerp_(g,1-fast_beta);slow.lerp_(g,.02)
            x=g.lerp(mix*fast+(1-mix)*slow,beta)
            x=x/(1.05*x.norm(dim=(-2,-1),keepdim=True)+1e-6)
            tall=x.shape[-2]>x.shape[-1]
            for a,b,c in MAPS:
                gram=x.mT@x if tall else x@x.mT
                poly=b*gram+c*(gram@gram)
                x=a*x+(x@poly if tall else poly@x)
            power=x.square().mean(bucket['axis'],keepdim=True)
            energy.lerp_(power,.1)
            gain=energy.clamp_min(1e-10).rsqrt()
            ratio=(power.sum((-2,-1),keepdim=True)/(power*gain.square()).sum((-2,-1),keepdim=True).clamp_min(1e-10)).sqrt()
            update=x*gain*ratio
            control=self.controls[3:].view(len(self.groups),3).index_select(0,bucket['index'])
            shape=(len(indices),)+(1,)*(p.ndim-1)
            rate=control[:,0].reshape(shape);decay=control[:,1].reshape(shape)
            p.addcmul_(p*(p*slow>=0),decay,value=-1)
            p.addcmul_(update,rate,value=-1)
            torch._foreach_copy_(params,list(p.unbind(0)))

    @torch.no_grad()
    def step(self,progress,multiplier):
        missing=any(p.grad is None for p in self.parameters)
        if self.gradient_scale!=1. and (missing or not self.scale_in_update):
            torch._foreach_div_([p.grad for p in self.parameters if p.grad is not None],
                               self.gradient_scale)
        if missing:
            self.fallbacks+=1
            self.base.step(progress,multiplier)
            return
        self.base.t+=1
        warm,cool=240/1194,50/1194
        beta=.85+.08*min(1.,progress/warm,max(0.,(1-progress)/cool))
        fast_beta,mix=(.85,.4385) if progress>=514/1194 else (beta,1.)
        values=[beta,fast_beta,mix]
        lr=.023*self.base.scale*multiplier
        for s in self.matrices:
            values.extend((lr*s['multiplier'],2.25*lr*lr*s['multiplier'],0.))
        lr=.008*self.base.scale*multiplier
        for s in self.adam:
            b=s['beta'];rate=lr*s['mul']
            correction=math.sqrt(1-b[1]**self.base.t)/(1-b[0]**self.base.t)
            values.extend((rate,rate*rate*s['wd'],correction))
        # Blocking transfer avoids reusing host storage before DMA completes.
        self.controls.copy_(torch.tensor(values,dtype=torch.float32),non_blocking=False)
        torch._foreach_copy_(self.grad_buffers,[p.grad for p in self.parameters])
        self.graph.replay()
