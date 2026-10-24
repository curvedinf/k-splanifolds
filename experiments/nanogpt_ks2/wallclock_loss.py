"""The reference training objective with optional FP16 output GEMM only.

Soft-capping, log-softmax and losses remain FP32. Evaluation retains the original
FP32 full-vocabulary head. Master weights and optimizer state remain FP32.
"""
import torch
from torch.nn import functional as F

from experiments.nanogpt_ks2 import run


def loss(model,tokens,prefix,step,steps,seed,windows,*,head_fp16=False,full_vocab=False,lr_end=.3,fused_head_loss=False):
    window,mtp,prefix_weight,lr,candidates=run.schedule(step,steps)
    if full_vocab:candidates=0
    if lr_end!=.3:
        progress=step/steps
        lr=1. if progress<.2 else 1.-(1.-lr_end)*(progress-.2)/.8
    for yarn,old,new in zip((model.yarn,model.yarn_paired_head,model.yarn_wide),windows,
                            (window[0],window[0],window[1])):
        if old!=new:yarn.apply_window(old,new)
    windows[:]=[window[0],window[0],window[1]]
    x=tokens[:-3]
    hidden=model(x,*window,run.documents(x))
    targets=tokens[1:len(x)+1];all_targets=tokens[1:];pref=prefix[all_targets]
    generator=torch.Generator(device=tokens.device).manual_seed(seed*100000+step)
    if candidates:
        ids=torch.unique(torch.cat((all_targets,pref[pref>=0],
                                    torch.randperm(50257,generator=generator,device=tokens.device)[:candidates])))
        weight=model.head_weight()[ids]
    else:ids,weight=None,model.head_weight()
    if head_fp16:
        with torch.autocast('cuda',dtype=torch.float16,cache_enabled=False):
            logits=F.linear(hidden,weight)
    else:logits=F.linear(hidden,weight)
    if fused_head_loss:
        from experiments.nanogpt_ks2.wallclock_softmax import softcap_loss
        columns=[tokens[1+shift:1+shift+len(x)] for shift in range(3)]
        prefix_target=prefix[targets];valid=prefix_target>=0
        columns.append(prefix_target.clamp_min(0))
        mapped=torch.stack([torch.searchsorted(ids,target) if ids is not None else target
                            for target in columns],dim=-1)
        # A disabled prefix term may point beyond the sampled vocabulary;
        # give it an in-range dummy index while preserving its zero weight.
        mapped[:,-1].clamp_(max=logits.shape[-1]-1)
        return softcap_loss(logits,mapped,valid,
                            mtp[1] if len(mtp)>1 else 0.,mtp[2] if len(mtp)>2 else 0.,prefix_weight),lr
    logits=logits.float()
    logits=23*torch.sigmoid((logits+5)/7.5)
    logp=F.log_softmax(logits,dim=-1)
    rows=torch.arange(len(x),device=x.device)
    result=logp.new_zeros(())
    for shift,coefficient in enumerate(mtp):
        target=tokens[1+shift:1+shift+len(x)]
        mapped=torch.searchsorted(ids,target) if ids is not None else target
        result=result-coefficient*logp[rows,mapped].mean()
    if prefix_weight:
        target=prefix[targets];valid=target>=0
        mapped=torch.searchsorted(ids,target.clamp_min(0)) if ids is not None else target.clamp_min(0)
        result=result-prefix_weight*(logp[rows,mapped]*valid).mean()
    return result,lr
