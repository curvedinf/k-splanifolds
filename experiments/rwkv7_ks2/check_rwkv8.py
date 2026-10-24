"""Paired architecture, ROSA gradient flow, and causality checks."""
import json
import torch
from torch.nn import functional as F
import run
from gpu_recurrence import recurrence
from side_history import HistoryModel
from rwkv8_rosa import MODES,RWKV8Model,make_optimizer,parameter_breakdown


def main():
    torch.set_num_threads(2)
    tokens=torch.randint(256,(2,8))
    baseline=HistoryModel(11,2048,64)
    with torch.no_grad():expected=baseline(tokens)
    original=baseline.state_dict()
    result={}
    cpu_recurrence=run.recurrence
    for mode,count in [('baseline',567552),('reduced_control',515328),('hybrid_matched',567296),('hybrid_full',619520)]:
        run.recurrence=cpu_recurrence
        model=RWKV8Model(11,mode)
        assert sum(p.numel() for p in model.parameters())==count,parameter_breakdown(model)
        sd=model.state_dict()
        for name,value in original.items():
            if '.attention.' in name and mode in ('reduced_control','hybrid_matched'):
                if sd[name].shape!=value.shape:
                    value=value[:sd[name].shape[0],:sd[name].shape[1]]
            assert torch.equal(sd[name],value),name
        with torch.no_grad():torch.testing.assert_close(model(tokens),expected,atol=0,rtol=0)
        model=model.cuda();run.recurrence=recurrence
        opt=make_optimizer(model,True)
        for _ in range(4):
            opt.zero_grad(set_to_none=True)
            logits=model(tokens.cuda())
            loss=F.cross_entropy(logits.flatten(0,1),tokens.roll(-1,1).cuda().flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step()
        if mode.startswith('hybrid'):
            for b in model.blocks:
                for name,p in b.rosa.named_parameters():
                    assert p.grad is not None and torch.isfinite(p.grad).all(),name
                assert all(getattr(b.rosa,name).weight.grad.abs().max()>0 for name in ('q','k','v','output'))
        changed=tokens.clone();changed[:,5:]=(changed[:,5:]+1)%256
        with torch.no_grad():
            torch.testing.assert_close(model(tokens.cuda())[:,:5],model(changed.cuda())[:,:5],atol=2e-6,rtol=2e-6)
        result[mode]=parameter_breakdown(model)
        del model,opt
    print(json.dumps(dict(status='PASS',arms=result,checks=['paired original state and exact initial CPU logits',
        'matched parameter counts','finite training and nonzero gradients through all ROSA QKV projections',
        'causal hybrid outputs']),indent=2))


if __name__=='__main__':main()
