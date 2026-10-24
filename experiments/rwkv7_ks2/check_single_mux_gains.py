"""Check paired weights, exact routing, dynamic normalization, and GPU gradients."""
import json
import torch
from torch.nn import functional as F
import run
from gpu_recurrence import recurrence
from single_mux import SingleMuxModel, SharedMux
from single_mux_gains import MuxGainModel, make_optimizer, parameter_breakdown


def main():
    torch.set_num_threads(2)
    tokens = torch.randint(256,(2,16))
    results = {}
    cpu_recurrence = run.recurrence
    for placement, count in [('output',542432),('input',544032),('dynamic',562736)]:
        run.recurrence = cpu_recurrence
        baseline = SingleMuxModel(11)
        model = MuxGainModel(11,placement=placement)
        assert parameter_breakdown(model)['total']==count
        assert sum(isinstance(m,SharedMux) for m in model.modules())==4
        for n,v in baseline.state_dict().items():
            q=model.state_dict()[n]
            if placement=='dynamic' and n.endswith('mux.weight'):q=q[:608]
            assert torch.equal(v,q),n
        if placement!='dynamic':
            torch.testing.assert_close(model(tokens),baseline(tokens),atol=0,rtol=0)
        for b in model.blocks:
            assert not hasattr(b.ffn,'down') and not hasattr(b.ffn,'up')
            x=torch.randn(2,4,b.history_width)
            if placement=='dynamic':
                joined=F.linear(x,b.mux.weight)
                z=b.mux.modulator(joined[...,608:])
                # Explicit sum of the eight scalar-weighted vectors, then ONE normalization.
                summed=sum(z[...,i,None]*b.mux.gain_vectors[i] for i in range(8))
                gains=summed/summed.square().mean(-1,keepdim=True).clamp_min(1e-8).sqrt()
                torch.testing.assert_close(gains.square().mean(-1),torch.ones_like(gains[...,0]))
                torch.testing.assert_close(b.mux(x),joined[...,:608]*gains,atol=2e-6,rtol=2e-5)
                assert (b.mux.weight[608:]!=0).all()
                assert not hasattr(b.mux.modulator,'down') and not hasattr(b.mux.modulator,'up')
            else:
                assert torch.equal(b.mux.gain,torch.ones_like(b.mux.gain))
                with torch.no_grad():b.mux.gain.copy_(torch.linspace(.8,1.2,b.mux.gain.numel()))
                expected=F.linear(x*b.mux.gain,b.mux.weight) if placement=='input' else F.linear(x,b.mux.weight)*b.mux.gain
                torch.testing.assert_close(b.mux(x),expected,atol=0,rtol=0)
                with torch.no_grad():b.mux.gain.fill_(1.)
        model=model.cuda();baseline=baseline.cuda();run.recurrence=recurrence
        logits=model(tokens.cuda())
        gpu_error=None
        if placement!='dynamic':
            expected=baseline(tokens.cuda())
            gpu_error=(logits-expected).abs().max().item()
            torch.testing.assert_close(logits,expected,atol=2e-6,rtol=2e-6)
            expected.square().mean().backward()
        logits.square().mean().backward()
        if placement!='dynamic':
            for n,p in baseline.named_parameters():
                q=dict(model.named_parameters())[n]
                if p.grad is None:assert q.grad is None
                else:torch.testing.assert_close(q.grad,p.grad,atol=2e-5,rtol=2e-4)
        new_params=[]
        for b in model.blocks:
            p=b.mux.gain_vectors if placement=='dynamic' else b.mux.gain
            new_params.append(p)
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0
            if placement=='output':
                assert p.grad[:512].abs().max()>0 and p.grad[512:].abs().max()>0
            if placement=='input':
                assert p.grad[:96].abs().max()>0
                for start in range(96,b.history_width,608):
                    assert p.grad[start:start+96].abs().max()>0
                    assert p.grad[start+96:start+608].abs().max()>0
            if placement=='dynamic':
                assert (p.grad.abs().sum(-1)>0).all()
                assert p.grad[:,:512].abs().max()>0 and p.grad[:,512:].abs().max()>0
                assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in b.mux.modulator.parameters())
                assert sum(q.grad.abs().sum().item() for q in b.mux.modulator.parameters())>0
        opt=make_optimizer(model,True)
        assert opt.param_groups[1]['lr_scale']==5.5
        assert abs(opt.param_groups[1]['weight_decay']-.01/5.5)<1e-12
        assert {id(p) for p in opt.param_groups[1]['params']}=={id(model.demux.weight),*(id(p) for p in new_params)}
        previous=[p.detach().clone() for p in new_params]
        opt.step()
        assert all(not torch.equal(p,q) for p,q in zip(new_params,previous))
        # Changing future tokens cannot affect earlier logits (including modulation).
        changed=tokens.cuda().clone();changed[:,8:]=(changed[:,8:]+1)%256
        with torch.no_grad():
            torch.testing.assert_close(model(tokens.cuda())[:,:8],model(changed)[:,:8],atol=2e-6,rtol=2e-6)
        results[placement]=dict(parameters=parameter_breakdown(model),initial_gpu_max_error=gpu_error)
        del model,baseline,opt
    print(json.dumps(dict(status='PASS',arms=results,checks=[
        'reference weights and main mux rows paired across all arms',
        'static gains preserve exact CPU initial logits; GPU parity within rounding tolerance',
        'one fixed mux per layer and no decoder projections',
        'dynamic gain is sum of eight KS2-scalar-weighted vectors, then unit-RMS normalization',
        'finite nonzero gain/extra-geometry gradients, optimizer grouping and parameter updates',
        'causal output for all three arms']),indent=2))


if __name__=='__main__':main()
