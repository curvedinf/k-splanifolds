"""Validate exact additive routing, identity gradients, pairing and GPU parity."""
import copy
import json
import torch
import run
from gpu_recurrence import recurrence
from raw_mux import RawMuxModel, parameter_breakdown
from raw_mux_skip import RawSkipModel


def main():
    torch.set_num_threads(2)
    baseline = RawMuxModel(11)
    model = RawSkipModel(11)
    assert parameter_breakdown(model)['total'] == 543136
    assert model.state_dict().keys() == baseline.state_dict().keys()
    assert all(torch.equal(v,baseline.state_dict()[n]) for n,v in model.state_dict().items())
    tokens = torch.randint(256,(2,8))
    for b in model.blocks[1:]: b.attention_mux.skip_scale = 0.
    torch.testing.assert_close(model(tokens),baseline(tokens),atol=0,rtol=0)
    for b in model.blocks[1:]: b.attention_mux.skip_scale = 1.
    banks, outputs, normalized, handles = [], [], [], []
    for b in model.blocks:
        handles.append(b.register_forward_pre_hook(lambda m,args:banks.append(args[0])))
        handles.append(b.register_forward_hook(lambda m,args,out:outputs.append(out)))
        handles.append(b.attention.register_forward_pre_hook(lambda m,args:normalized.append(args[0])))
    logits = model(tokens)
    for h in handles: h.remove()
    for i,b in enumerate(model.blocks):
        view = torch.nn.functional.linear(banks[i],b.attention_mux.weight)
        combined = view if not i else view + outputs[i-1][0]
        torch.testing.assert_close(normalized[i],b.norm1(combined),atol=0,rtol=0)
        # RWKV output projections start at zero, so post-attention == combined.
        torch.testing.assert_close(outputs[i][0],combined,atol=0,rtol=0)
        outputs[i][0].retain_grad()
    logits.square().mean().backward()
    assert all(out[0].grad.abs().max()>0 for out in outputs)
    # Suppress the mux to isolate the coefficient-one identity path.
    mux = copy.deepcopy(model.blocks[-1].attention_mux)
    mux.weight.zero_()
    bank = torch.randn(2,8,1920,requires_grad=True)
    mux(bank).sum().backward()
    expected = torch.zeros_like(bank); expected[...,1312:1408] = 1
    torch.testing.assert_close(bank.grad,expected,atol=0,rtol=0)
    changed = tokens.clone(); changed[:,5:] = (changed[:,5:]+1)%256
    torch.testing.assert_close(model(tokens)[:,:5],model(changed)[:,:5],atol=0,rtol=0)
    with torch.no_grad():
        for b in model.blocks: b.attention.output.weight.normal_(std=.01)
    model.zero_grad(set_to_none=True)
    model = model.cuda(); other = copy.deepcopy(model)
    expected = model(tokens.cuda()); expected.square().mean().backward()
    run.recurrence = recurrence
    actual = other(tokens.cuda()); actual.square().mean().backward()
    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-4)
    for (name,p),(_,q) in zip(model.named_parameters(),other.named_parameters()):
        if p.grad is None: assert q.grad is None
        else: torch.testing.assert_close(q.grad,p.grad,atol=2e-4,rtol=2e-3,msg=lambda msg:name+': '+msg)
    print(json.dumps(dict(status='PASS',parameters=parameter_breakdown(model),checks=[
        'all parameters and buffers exactly paired with raw-mux baseline',
        'zero skip reproduces baseline logits exactly',
        'previous post-attention state plus mux before normalization and residual addition',
        'coefficient-one direct gradient path', 'causal forward',
        'GPU recurrence forward and all gradients match reference']),indent=2))


if __name__ == '__main__':
    main()
