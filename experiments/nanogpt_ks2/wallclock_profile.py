"""Phase timing diagnostic for the full-vocabulary training stage.

Synchronizing each phase prevents overlap, so these are bottleneck diagnostics,
not replacements for complete-run convergence timings.
"""
import argparse
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from experiments.nanogpt_ks2.wallclock import make_model
from experiments.nanogpt_ks2.run import documents


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--gpu',type=int,default=3)
    args=ap.parse_args();cfg=json.loads(args.config.read_text())
    torch.cuda.set_device(args.gpu);torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    settings=SimpleNamespace(device=f'cuda:{args.gpu}',sequence=cfg.get('sequence',4096),width=96,
                              lr_scale=cfg.get('lr_scale',.25))
    model,opt=make_model(cfg,11,settings)
    old=(16,16,48)
    for window in ((48,112),(80,176)):
        new=(window[0],window[0],window[1])
        for yarn,a,b in zip((model.yarn,model.yarn_paired_head,model.yarn_wide),old,new):
            yarn.apply_window(a,b)
        old=new
    tokens=torch.randint(0,50257,(settings.sequence+1,),device=settings.device)
    ids=documents(tokens[:-1]);rows=torch.arange(settings.sequence,device=settings.device)
    labels=tokens[1:];targets=labels[:,None].expand(-1,4).contiguous()
    valid=torch.zeros_like(labels,dtype=torch.bool)
    samples=[]
    def stamp():
        torch.cuda.synchronize();return time.perf_counter()
    for iteration in range(13):
        opt.zero_grad();times=[stamp()]
        hidden=model(tokens[:-1],80,176,ids);times.append(stamp())
        detached=hidden.detach().requires_grad_()
        if cfg.get('head_fp16',False):
            with torch.autocast('cuda',dtype=torch.float16,cache_enabled=False):
                logits=F.linear(detached,model.head_weight())
        else:logits=F.linear(detached,model.head_weight())
        if cfg.get('fused_head_loss',False):
            from experiments.nanogpt_ks2.wallclock_softmax import softcap_loss
            value=softcap_loss(logits,targets,valid)
        else:
            capped=23*torch.sigmoid((logits.float()+5)/7.5)
            value=-F.log_softmax(capped,-1)[rows,labels].mean()
        value=value*cfg.get('gradient_scale',1.)
        times.append(stamp())
        value.backward();times.append(stamp())
        hidden.backward(detached.grad);times.append(stamp())
        opt.step(.95,.34375);times.append(stamp())
        if iteration>=3:samples.append([1000*(b-a) for a,b in zip(times,times[1:])])
        del hidden,detached,logits,value
        if not cfg.get('fused_head_loss',False):del capped
    names=['model_forward_ms','head_forward_loss_ms','head_backward_ms','model_backward_ms','optimizer_ms']
    result=dict(candidate=cfg,phase_medians={name:statistics.median(x[i] for x in samples)
                                            for i,name in enumerate(names)},samples_ms=samples,
                note='Fresh-model diagnostic, full-vocabulary final stage; phase synchronization prevents overlap.')
    args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['phase_medians']))


if __name__=='__main__':main()
