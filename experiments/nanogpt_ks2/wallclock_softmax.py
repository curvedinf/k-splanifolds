"""Fused FP32 softcap / multi-target cross-entropy with recomputed backward."""
import torch
import triton as tr
import triton.language as tl


@tr.jit(do_not_specialize=['V','C1','C2','CP'])
def _forward(X,TARGET,VALID,LOSS,LSE,V,C1,C2,CP,BLOCK:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,BLOCK)
    x=tl.load(X+row*V+col,col<V,other=0.).to(tl.float32)
    y=23./(1.+tl.exp(-(x+5.)/7.5))
    y=tl.where(col<V,y,float('-inf'))
    maximum=tl.max(y,0)
    lse=maximum+tl.log(tl.sum(tl.exp(y-maximum),0))
    valid=tl.load(VALID+row).to(tl.float32)
    result=(1.+C1+C2+CP*valid)*lse
    for shift in tl.static_range(4):
        target=tl.load(TARGET+row*4+shift)
        value=tl.load(X+row*V+target).to(tl.float32)
        capped=23./(1.+tl.exp(-(value+5.)/7.5))
        if shift==0:coefficient=1.
        elif shift==1:coefficient=C1
        elif shift==2:coefficient=C2
        else:coefficient=CP*valid
        result=result-coefficient*capped
    tl.store(LOSS+row,result);tl.store(LSE+row,lse)


@tr.jit(do_not_specialize=['V','N','C1','C2','CP'])
def _backward(X,TARGET,VALID,LSE,GRAD,DX,V,N,C1,C2,CP,BLOCK:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,BLOCK)
    x=tl.load(X+row*V+col,col<V,other=0.).to(tl.float32)
    sigmoid=1./(1.+tl.exp(-(x+5.)/7.5))
    y=23.*sigmoid;lse=tl.load(LSE+row)
    valid=tl.load(VALID+row).to(tl.float32)
    dy=(1.+C1+C2+CP*valid)*tl.exp(y-lse)
    for shift in tl.static_range(4):
        target=tl.load(TARGET+row*4+shift)
        if shift==0:coefficient=1.
        elif shift==1:coefficient=C1
        elif shift==2:coefficient=C2
        else:coefficient=CP*valid
        dy=dy-tl.where(col==target,coefficient,0.)
    scale=tl.load(GRAD)/N
    dx=dy*(23./7.5)*sigmoid*(1.-sigmoid)*scale
    tl.store(DX+row*V+col,dx,col<V)


class _SoftcapLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx,logits,targets,valid,c1,c2,cp):
        if not logits.is_contiguous() or not targets.is_contiguous():
            raise ValueError('Contiguous logits and targets required')
        n,v=logits.shape
        rows=torch.empty(n,device=logits.device,dtype=torch.float32)
        lse=torch.empty_like(rows)
        block=tr.next_power_of_2(v);warps=16 if block>=32768 else 8
        _forward[(n,)](logits,targets,valid,rows,lse,v,c1,c2,cp,block,
                       num_warps=warps,enable_fp_fusion=False)
        ctx.save_for_backward(logits,targets,valid,lse)
        ctx.settings=(c1,c2,cp,block,warps)
        return rows.mean()

    @staticmethod
    def backward(ctx,grad):
        logits,targets,valid,lse=ctx.saved_tensors
        c1,c2,cp,block,warps=ctx.settings;n,v=logits.shape
        result=torch.empty_like(logits)
        _backward[(n,)](logits,targets,valid,lse,grad,result,v,n,c1,c2,cp,block,
                        num_warps=warps,enable_fp_fusion=False)
        return result,None,None,None,None,None


def softcap_loss(logits,targets,valid,c1=0.,c2=0.,cp=0.):
    """Targets are [tokens, 4]: three next-token shifts plus the prefix target."""
    return _SoftcapLoss.apply(logits,targets,valid,float(c1),float(c2),float(cp))
