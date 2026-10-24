"""Validate raw-coordinate routing, pre-RWKV muxes and the final-only demux."""
import copy
import json
import torch
import run
from gpu_recurrence import recurrence
from raw_mux import RawMuxModel, parameter_breakdown
from side_stream import SideModel
from spread_projection import FixedOutput


def main():
    torch.set_num_threads(2)
    model = RawMuxModel(11)
    baseline = SideModel(11, 2048, 64)
    assert parameter_breakdown(model)['total'] == 543136
    assert sum(isinstance(m, FixedOutput) for m in model.modules()) == 1
    assert torch.equal(model.demux.weight, torch.ones(96))
    for i, (b, ref) in enumerate(zip(model.blocks, baseline.blocks)):
        assert not hasattr(b.ffn, 'up')
        assert b.history_width == 96 + i * 608
        assert b.ffn_input_width == 192 + i * 608
        for name, value in b.ffn.named_parameters():
            assert name.startswith('geo_')
            assert torch.equal(value, dict(ref.ffn.named_parameters())[name])
        assert all(torch.equal(v, ref.attention.state_dict()[n]) for n,v in b.attention.state_dict().items())
        for mux, charts in ((b.attention_mux,12), (b.ffn.down,64)):
            codes = mux.weight.double().reshape(charts,8,-1)
            assert (codes != 0).all() and torch.unique(codes.flatten(1), dim=0).shape[0] == charts
            torch.testing.assert_close(codes @ codes.transpose(-1,-2),
                                      torch.eye(8,dtype=torch.float64).expand(charts,8,8), atol=2e-6, rtol=2e-6)
    tokens = torch.randint(256, (2,8))
    banks, outputs, attention_inputs, ffn_inputs, handles = [], [], [], [], []
    for b in model.blocks:
        handles.append(b.register_forward_pre_hook(lambda m,args: banks.append(args[0])))
        handles.append(b.register_forward_hook(lambda m,args,out: outputs.append(out)))
        handles.append(b.attention.register_forward_pre_hook(lambda m,args: attention_inputs.append(args[0])))
        handles.append(b.norm2.register_forward_pre_hook(lambda m,args: ffn_inputs.append(args[0])))
    logits = model(tokens)
    for handle in handles: handle.remove()
    for i, b in enumerate(model.blocks):
        torch.testing.assert_close(attention_inputs[i], b.norm1(b.attention_mux(banks[i])), atol=0, rtol=0)
        torch.testing.assert_close(ffn_inputs[i], torch.cat((banks[i],outputs[i][0]),-1), atol=0, rtol=0)
        assert outputs[i][1].shape == (2,8,512)
        if i:
            expected = torch.cat((banks[i-1],outputs[i-1][0],outputs[i-1][1]),-1)
            torch.testing.assert_close(banks[i], expected, atol=0, rtol=0)
        for tensor in outputs[i][:2]: tensor.retain_grad()
    torch.testing.assert_close(logits, model.head(model.output_norm(model.demux(outputs[-1][1]))), atol=0, rtol=0)
    logits.square().mean().backward()
    assert all(t.grad is not None and t.grad.abs().max()>0 for out in outputs for t in out[:2])
    for b in model.blocks:
        assert any(p.grad.abs().max()>0 for p in b.ffn.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    # Direct sensitivity of the final block to every prior attention/KS2 slice.
    bank = banks[-1].detach().requires_grad_()
    model.zero_grad(set_to_none=True)
    model.blocks[-1](bank,None)[1].square().mean().backward()
    for start,end in ((0,96),(96,192),(192,704),(704,800),(800,1312),(1312,1408),(1408,1920)):
        assert bank.grad[...,start:end].abs().max()>0
    altered = tokens.clone(); altered[:,5:] = (altered[:,5:]+1)%256
    torch.testing.assert_close(model(tokens)[:,:5],model(altered)[:,:5],atol=0,rtol=0)
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
        'exact raw-output history and pre-RWKV mux routing', 'one final demux only; no head bypass',
        'all-input coverage and per-view orthonormal rows', 'paired geometry/attention initialization',
        'all history slices and all four geometries receive gradients', 'causal forward',
        'GPU forward and all gradients match reference']),indent=2))


if __name__ == '__main__':
    main()
