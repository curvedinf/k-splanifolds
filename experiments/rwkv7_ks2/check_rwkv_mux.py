"""Check isolated RWKV replacement coverage, fixed codes, initialization and gradients."""
import copy
import json
import torch
from torch import nn
from torch.nn import functional as F
import run
from gpu_recurrence import recurrence
from side_history import HistoryModel
from rwkv_mux import MODES,PROJECTIONS,HadamardMux,RWKVMuxModel,make_optimizer,parameter_breakdown


def main():
    torch.set_num_threads(2)
    tokens=torch.randint(256,(2,12))
    baseline=HistoryModel(11,2048,64)
    original=baseline.state_dict()
    cpu_recurrence=run.recurrence
    with torch.no_grad():initial=baseline(tokens)
    base_count=sum(p.numel() for p in baseline.parameters())
    assert base_count==567552
    assert sum(isinstance(m,nn.Linear) for b in baseline.blocks for m in b.attention.modules())==46
    report={}
    for mode in MODES:
        run.recurrence=cpu_recurrence
        model=RWKVMuxModel(11,mode)
        replaced={f"blocks.{r['layer']}.attention.{r['projection']}.weight" for r in model.replacements}
        for name,value in original.items():
            if name not in replaced:assert torch.equal(model.state_dict()[name],value),name
        with torch.no_grad():torch.testing.assert_close(model(tokens),initial,atol=0,rtol=0)
        muxes=[m for b in model.blocks for m in b.attention.modules() if isinstance(m,HadamardMux)]
        expected=46 if mode=='all' else (0 if mode=='baseline' else (3 if mode.startswith('value_control.') else 4))
        assert len(muxes)==expected
        assert sum(isinstance(m,nn.Linear) for b in model.blocks for m in b.attention.modules())==46-expected
        assert isinstance(model.head,nn.Linear)
        for r in model.replacements:
            prefix=f"blocks.{r['layer']}.attention.{r['projection']}"
            m=model.get_submodule(prefix)
            w=original[prefix+'.weight']
            torch.testing.assert_close(m.gain,w.square().sum(1).sqrt(),atol=0,rtol=0)
            assert (m.matrix!=0).all()
            torch.testing.assert_close(m.matrix.square().sum(1),torch.ones(m.out_features))
            assert torch.linalg.matrix_rank(m.matrix.double()).item()==min(m.in_features,m.out_features),prefix
            grouped=m.matrix.double().reshape(-1,8,m.in_features)
            torch.testing.assert_close(grouped@grouped.transpose(-1,-2),torch.eye(8,dtype=torch.float64).expand(len(grouped),8,8),atol=2e-6,rtol=2e-6)
            x=torch.randn(2,3,m.in_features)
            torch.testing.assert_close(m(x),F.linear(x,m.matrix)*m.gain,atol=0,rtol=0)
        count=sum(p.numel() for p in model.parameters())
        assert count==base_count-sum(r['removed_dense_parameters']-r['added_gains'] for r in model.replacements)
        if mode=='all':assert count==306944
        model=model.cuda();run.recurrence=recurrence
        opt=make_optimizer(model,True)
        assert opt.param_groups[0]['lr_scale']==1. and opt.param_groups[0]['weight_decay']==.01
        assert opt.param_groups[1]['lr_scale']==5.5
        group0={id(p) for p in opt.param_groups[0]['params']}
        assert all(id(m.gain) in group0 for m in muxes)
        before=[m.gain.detach().clone() for m in muxes]
        # Zero residual outputs delay upstream gradients; allow the paths to open naturally.
        for _ in range(4):
            opt.zero_grad(set_to_none=True)
            logits=model(tokens.cuda())
            loss=F.cross_entropy(logits.flatten(0,1),tokens.roll(-1,1).cuda().flatten())
            assert torch.isfinite(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step()
        for m,old in zip(muxes,before):
            assert m.gain.grad is not None and torch.isfinite(m.gain.grad).all() and m.gain.grad.abs().max()>0,mode
            assert not torch.equal(m.gain,old),mode
        changed=tokens.clone();changed[:,7:]=(changed[:,7:]+1)%256
        with torch.no_grad():
            torch.testing.assert_close(model(tokens.cuda())[:,:7],model(changed.cuda())[:,:7],atol=2e-6,rtol=2e-6)
        if mode=='all':
            reference=copy.deepcopy(model)
            model.zero_grad(set_to_none=True);reference.zero_grad(set_to_none=True)
            run.recurrence=cpu_recurrence
            expected_logits=reference(tokens.cuda());expected_logits.square().mean().backward()
            run.recurrence=recurrence
            actual=model(tokens.cuda());actual.square().mean().backward()
            torch.testing.assert_close(actual,expected_logits,atol=2e-5,rtol=2e-4)
            for (name,p),(_,q) in zip(model.named_parameters(),reference.named_parameters()):
                if p.grad is None:assert q.grad is None
                else:torch.testing.assert_close(p.grad,q.grad,atol=2e-4,rtol=2e-3,msg=lambda msg:name+': '+msg)
            del reference
        report[mode]=dict(parameters=count,replacements=len(muxes))
        print(json.dumps(dict(event='checked',mode=mode,**report[mode])),flush=True)
        del model,opt
    print(json.dumps(dict(status='PASS',arms=report,checks=[
        'exact unchanged baseline state and CPU initial logits for every arm',
        'all 46 RWKV linears covered; each isolated arm replaces only its named projection type',
        'all-input fixed Hadamard codes, unit row norms, orthonormal rank-8 groups and full attainable matrix rank',
        'original row norms and zero-initialized paths preserved',
        'base LR for RWKV gains; unchanged decoder gain optimizer settings',
        'finite training, nonzero gradients and actual gain updates after zero paths open',
        'causal output in every arm; all-mux GPU recurrence and gradients agree with reference']),indent=2))


if __name__=='__main__':main()
