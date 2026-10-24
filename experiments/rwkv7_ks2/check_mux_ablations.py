"""Verify each mux hypothesis changes only its designated route or parameters."""
import copy
import json
import torch
import run
from gpu_recurrence import recurrence
from raw_mux import RawMuxModel
from side_history import HistoryModel
from mux_ablations import build_model, parameter_breakdown, normalize_sources, make_optimizer


def main():
    torch.set_num_threads(2)
    original_recurrence = run.recurrence
    tokens = torch.randint(256,(2,8))
    base = RawMuxModel(11)
    wrapped = build_model(11)
    assert all(torch.equal(v,wrapped.state_dict()[n]) for n,v in base.state_dict().items())
    torch.testing.assert_close(base(tokens),wrapped(tokens),atol=0,rtol=0)
    old = HistoryModel(11,2048,64); new = build_model(11,mode='demuxed_history')
    assert all(torch.equal(v,new.state_dict()[n]) for n,v in old.state_dict().items())
    del old,new
    changed = build_model(11,code_seed=103416)
    assert all(torch.equal(p,dict(changed.named_parameters())[n]) for n,p in base.named_parameters())
    assert not torch.equal(base.blocks[0].attention_mux.weight,changed.blocks[0].attention_mux.weight)
    rows=[]
    for mode in ('head_skip','output_gains','persistent_state','source_norm'):
        run.recurrence = original_recurrence
        model=build_model(11,mode=mode)
        expected_count=545184 if mode=='output_gains' else 543136
        assert parameter_breakdown(model)['total']==expected_count
        for n,v in base.state_dict().items(): assert torch.equal(v,model.state_dict()[n]),(mode,n)
        assert all(type(b.attention_mux).__name__=='FixedInput' for b in model.blocks)
        banks,outputs,attn_inputs,handles=[],[],[],[]
        for b in model.blocks:
            handles.append(b.register_forward_pre_hook(lambda m,args:banks.append(args[0])))
            handles.append(b.register_forward_hook(lambda m,args,out:outputs.append(out)))
            handles.append(b.attention.register_forward_pre_hook(lambda m,args:attn_inputs.append(args[0])))
        logits=model(tokens)
        for h in handles:h.remove()
        state=None
        for i,b in enumerate(model.blocks):
            inputs=normalize_sources(banks[i]) if mode=='source_norm' else banks[i]
            torch.testing.assert_close(attn_inputs[i],b.norm1(b.attention_mux(inputs)),atol=0,rtol=0)
            raw=outputs[i][1]
            if mode=='persistent_state':state=raw if state is None else state+raw
            else:state=raw
            if i<3:
                torch.testing.assert_close(banks[i+1],torch.cat((banks[i],outputs[i][0],state),-1),atol=0,rtol=0)
        readout=model.demux(state)
        if mode=='head_skip':readout=readout+outputs[-1][0]
        torch.testing.assert_close(logits,model.head(model.output_norm(readout)),atol=0,rtol=0)
        logits.square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert all(any(p.grad.abs().max()>0 for p in b.ffn.parameters()) for b in model.blocks)
        if mode=='output_gains':
            assert all(torch.equal(b.raw_gain,torch.full((512,),.05)) and b.raw_gain.grad.abs().max()>0 for b in model.blocks)
            opt=make_optimizer(model,True)
            assert opt.param_groups[1]['lr_scale']==5.5 and len(opt.param_groups[1]['params'])==5
        altered=tokens.clone();altered[:,5:]=(altered[:,5:]+1)%256
        torch.testing.assert_close(model(tokens)[:,:5],model(altered)[:,:5],atol=0,rtol=0)
        with torch.no_grad():
            for b in model.blocks:b.attention.output.weight.normal_(std=.01)
        model.zero_grad(set_to_none=True)
        model=model.cuda(); other=copy.deepcopy(model)
        expected=model(tokens.cuda());expected.square().mean().backward()
        run.recurrence=recurrence
        actual=other(tokens.cuda());actual.square().mean().backward()
        torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-4)
        for (name,p),(_,q) in zip(model.named_parameters(),other.named_parameters()):
            if p.grad is None:assert q.grad is None
            else:torch.testing.assert_close(q.grad,p.grad,atol=2e-4,rtol=2e-3,msg=lambda msg:name+': '+msg)
        rows.append(dict(mode=mode,parameters=expected_count,status='PASS'))
        del model,other
    print(json.dumps(dict(checks='Exact reference reproduction; independently changed routes; no attention skips; paired parameters; finite geometry/gain gradients; causality; GPU forward/all-gradient parity',arms=rows),indent=2))


if __name__=='__main__':main()
