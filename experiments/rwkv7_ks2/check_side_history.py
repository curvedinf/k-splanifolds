"""Check history capture, coverage, gradients, causality and GPU recurrence parity."""
import copy
import json
import torch
from side_stream import SideModel, parameter_breakdown
from side_history import HistoryModel
import run
from gpu_recurrence import recurrence


def main():
    torch.set_num_threads(2)
    baseline = SideModel(11, 2048, 64)
    model = HistoryModel(11, 2048, 64)
    tokens = torch.randint(256, (2, 8))
    torch.testing.assert_close(model(tokens), baseline(tokens), atol=0, rtol=0)
    assert parameter_breakdown(model)['total'] == 567552
    for i, (b, ref) in enumerate(zip(model.blocks, baseline.blocks)):
        assert b.input_width == 2144 + 192 * i
        assert b.ffn.down.weight.shape == (512, b.input_width)
        codes = b.ffn.down.weight.double().reshape(64, 8, -1)
        assert (codes != 0).all() and torch.unique(codes.flatten(1), dim=0).shape[0] == 64
        torch.testing.assert_close(codes @ codes.transpose(-1, -2),
                                  torch.eye(8, dtype=torch.float64).expand(64, 8, 8), atol=2e-6, rtol=2e-6)
        for name, value in b.ffn.state_dict().items():
            if name != 'down.weight':
                assert torch.equal(value, ref.ffn.state_dict()[name]), name
        assert all(torch.equal(v, ref.attention.state_dict()[n]) for n,v in b.attention.state_dict().items())
        assert torch.equal(b.mix[:2144], ref.mix)
    with torch.no_grad():
        for b in model.blocks:
            b.ffn.up.weight.fill_(.05)
            b.attention.output.weight.normal_(std=.01)
    captures, inputs, handles = [], [], []
    for b in model.blocks:
        handles.append(b.norm2.register_forward_pre_hook(lambda m, args: inputs.append(args[0])))
        handles.append(b.register_forward_hook(lambda m, args, out: captures.append((out[3], out[0]))))
    logits = model(tokens)
    for h in handles:
        h.remove()
    for i in range(1, 4):
        expected = torch.cat([v for pair in captures[:i] for v in pair], dim=-1)
        torch.testing.assert_close(inputs[i][..., 2144:], expected, atol=0, rtol=0)
        assert inputs[i].grad_fn is not None
    logits.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    # Independent leaves prove that every history slice reaches the final FFN output.
    history = [v.detach().requires_grad_() for pair in captures[:3] for v in pair]
    model.zero_grad(set_to_none=True)
    out = model.blocks[3](torch.randn(2,8,96), torch.randn(2,8,2048), None, history)
    out[0].square().mean().backward()
    assert all(v.grad is not None and v.grad.abs().max() > 0 for v in history)
    altered = tokens.clone()
    altered[:, 5:] = (altered[:, 5:] + 1) % 256
    torch.testing.assert_close(model(tokens)[:, :5], model(altered)[:, :5], atol=0, rtol=0)
    model.zero_grad(set_to_none=True)
    model = model.cuda()
    other = copy.deepcopy(model)
    expected = model(tokens.cuda())
    expected.square().mean().backward()
    run.recurrence = recurrence
    actual = other(tokens.cuda())
    actual.square().mean().backward()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    for (name,p), (_,q) in zip(model.named_parameters(), other.named_parameters()):
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(q.grad, p.grad, atol=2e-4, rtol=2e-3,
                                       msg=lambda msg: name + ': ' + msg)
    print(json.dumps(dict(status='PASS', parameters=parameter_breakdown(model), checks=[
        'paired geometry/output/attention initialization and exact initial logits',
        'distinct all-input Hadamard codes with orthonormal rows',
        'separate post-attention/post-FFN residuals from strictly previous layers',
        'nonzero gradients to every history slice', 'causal forward',
        'GPU recurrence forward and all gradients match reference']), indent=2))


if __name__ == '__main__':
    main()
