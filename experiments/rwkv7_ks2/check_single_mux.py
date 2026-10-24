"""Check one mux per layer, geometry-only decoder, and live final attention path."""
import copy
import json
import torch
import run
from gpu_recurrence import recurrence
from raw_mux import RawMuxModel, parameter_breakdown
from single_mux import SingleMuxModel, SharedMux, hadamard96
from spread_projection import FixedInput, FixedOutput


def main():
    torch.set_num_threads(2)
    model = SingleMuxModel(11)
    base = RawMuxModel(11)
    assert parameter_breakdown(model)['total']==540000
    assert sum(isinstance(m,SharedMux) for m in model.modules())==4
    assert sum(isinstance(m,FixedInput) for m in model.modules())==0
    assert sum(isinstance(m,FixedOutput) for m in model.modules())==1
    torch.testing.assert_close(hadamard96().double() @ hadamard96().double().T,
                              torch.eye(96,dtype=torch.float64),atol=2e-6,rtol=2e-6)
    for i,(b,ref) in enumerate(zip(model.blocks,base.blocks)):
        assert not hasattr(b.ffn,'down') and not hasattr(b.ffn,'up')
        w=b.mux.weight.double()
        assert w.shape==(608,96+i*608) and (w!=0).all()
        torch.testing.assert_close(w.square().sum(1),torch.ones(608,dtype=torch.float64),atol=2e-6,rtol=2e-6)
        assert torch.linalg.matrix_rank(w[512:]).item()==96
        for name,value in b.ffn.named_parameters():
            assert torch.equal(value,dict(ref.ffn.named_parameters())[name])
        assert all(torch.equal(v,ref.attention.state_dict()[n]) for n,v in b.attention.state_dict().items())
    banks,muxes,outputs,attn_inputs,ks2_inputs,handles=[],[],[],[],[],[]
    for b in model.blocks:
        handles.append(b.register_forward_pre_hook(lambda m,args:banks.append(args[0])))
        handles.append(b.mux.register_forward_hook(lambda m,args,out:muxes.append(out)))
        handles.append(b.attention.register_forward_pre_hook(lambda m,args:attn_inputs.append(args[0])))
        handles.append(b.ffn.register_forward_pre_hook(lambda m,args:ks2_inputs.append(args[0])))
        handles.append(b.register_forward_hook(lambda m,args,out:outputs.append(out)))
    tokens=torch.randint(256,(2,8))
    logits=model(tokens)
    for h in handles:h.remove()
    assert len(muxes)==4
    for i,b in enumerate(model.blocks):
        torch.testing.assert_close(attn_inputs[i],b.norm1(muxes[i][...,512:]),atol=0,rtol=0)
        coords=muxes[i][...,:512]
        mixed=coords+(torch.nn.functional.pad(coords[:,:-1],(0,0,1,0))-coords)*b.mix
        torch.testing.assert_close(ks2_inputs[i],mixed,atol=0,rtol=0)
        if i:
            torch.testing.assert_close(banks[i],torch.cat((banks[i-1],outputs[i-1][0],outputs[i-1][1]),-1),atol=0,rtol=0)
        outputs[i][0].retain_grad();outputs[i][1].retain_grad()
    torch.testing.assert_close(logits,model.head(model.output_norm(model.demux(torch.cat((outputs[-1][1],outputs[-1][0]),-1)))),atol=0,rtol=0)
    logits.square().mean().backward()
    assert all(out[j].grad.abs().max()>0 for out in outputs for j in (0,1))
    assert model.blocks[-1].attention.output.weight.grad.abs().max()>0
    assert all(any(p.grad.abs().max()>0 for p in b.ffn.parameters()) for b in model.blocks)
    altered=tokens.clone();altered[:,5:]=(altered[:,5:]+1)%256
    torch.testing.assert_close(model(tokens)[:,:5],model(altered)[:,:5],atol=0,rtol=0)
    with torch.no_grad():
        for b in model.blocks:b.attention.output.weight.normal_(std=.01)
    model.zero_grad(set_to_none=True)
    model=model.cuda();other=copy.deepcopy(model)
    expected=model(tokens.cuda());expected.square().mean().backward()
    run.recurrence=recurrence
    actual=other(tokens.cuda());actual.square().mean().backward()
    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-4)
    for (name,p),(_,q) in zip(model.named_parameters(),other.named_parameters()):
        if p.grad is None:assert q.grad is None
        else:torch.testing.assert_close(q.grad,p.grad,atol=2e-4,rtol=2e-3,msg=lambda msg:name+': '+msg)
    print(json.dumps(dict(status='PASS',parameters=parameter_breakdown(model),checks=[
        'exactly one shared mux call per layer; zero decoder projections',
        'one full-rank 96D RWKV view and 64 rank-8 KS2 views; all-input coverage',
        'exact history/split routing and paired attention/geometry initialization',
        'final demux consumes full 608D residual; final RWKV and every KS2 receive gradients',
        'causal forward; GPU recurrence forward/all-gradient parity']),indent=2))


if __name__=='__main__':main()
