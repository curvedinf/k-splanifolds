"""Validate side routing, paired initialization, useful gradients and GPU parity."""
import copy
import json
import torch
from torch.nn import functional as F
import run
from side_stream import SideModel, parameter_breakdown
from gpu_recurrence import recurrence


def main():
    torch.set_num_threads(2)
    torch.manual_seed(12345)
    tokens = torch.randint(256, (2, 16))
    reference = run.Model(11, 'mlp', width=96, layers=4, fused_cpu=False)
    zero = SideModel(11, 0)
    rows = []
    for side_width in (0, 32, 96, 160):
        model = SideModel(11, side_width)
        assert len(model.blocks) == 4
        assert model.embedding.weight.shape == (256, 96) and model.head.weight.shape == (256, 96)
        for i, block in enumerate(model.blocks):
            assert block.attention.width == 96 and block.norm1.normalized_shape == (96,)
            assert block.norm2.normalized_shape == (96 + side_width,)
            assert not hasattr(block, 'key') and not hasattr(block, 'value')
            assert (block.ffn.num_subspaces, block.ffn.subspace_rank) == (32, 8)
            codes = block.ffn.down.weight.reshape(32, 8, 96 + side_width)
            assert (codes != 0).all()
            assert torch.unique(codes.flatten(1), dim=0).shape[0] == 32
            for code in codes:
                torch.testing.assert_close(code @ code.T, torch.eye(8), atol=2e-6, rtol=2e-6)
            for name, p in block.attention.state_dict().items():
                assert torch.equal(p, zero.blocks[i].attention.state_dict()[name])
            for name, p in block.ffn.named_parameters():
                if name.startswith('geo_'):
                    assert torch.equal(p, dict(zero.blocks[i].ffn.named_parameters())[name])
            assert torch.equal(block.mix[:96], zero.blocks[i].mix)
        torch.testing.assert_close(model(tokens), reference(tokens), atol=0, rtol=0)
        breakdown = parameter_breakdown(model)
        assert breakdown['total'] == 426368 + 16 * side_width
        assert breakdown['rwkv'] == 269568 and breakdown['embedding_and_head'] == 49152
        assert breakdown['ks2_geometry'] == 104960
        rows.append(dict(side_width=side_width, **breakdown))
    model = SideModel(11, 32)
    side = torch.randn(2, 16, 32, requires_grad=True)
    # With FFNs disabled, there is no side -> attention/head route.
    torch.testing.assert_close(model(tokens, side), model(tokens), atol=0, rtol=0)
    with torch.no_grad():
        for block in model.blocks:
            block.ffn.up.weight.fill_(.05)
            block.attention.output.weight.normal_(std=.01)
    main = torch.randn(2, 16, 96)
    observed = []
    handle = model.blocks[0].attention.register_forward_pre_hook(lambda m, args: observed.append(args[0].detach().clone()))
    model.blocks[0](main, torch.zeros_like(side), None)
    model.blocks[0](main, side, None)
    handle.remove()
    assert torch.equal(observed[0], observed[1])
    logits = model(tokens, side)
    assert not torch.allclose(logits, model(tokens))
    F.cross_entropy(logits.flatten(0, 1), tokens.flatten()).backward()
    assert side.grad.abs().max() > 0
    for block in model.blocks[:-1]:
        assert block.ffn.up.weight.grad[96:].abs().max() > 0
    assert model.blocks[-1].ffn.up.weight.grad[96:].count_nonzero() == 0
    # The terminal side update is discarded, so changing only it cannot
    # directly change logits. Its gain slots remain counted in parameters.
    with torch.no_grad():
        before = model(tokens).clone()
        model.blocks[-1].ffn.up.weight[96:].fill_(100.)
        torch.testing.assert_close(model(tokens), before, atol=0, rtol=0)
    model.zero_grad(set_to_none=True)
    model = model.cuda()
    other = copy.deepcopy(model)
    gpu_tokens = tokens.cuda()
    expected = model(gpu_tokens)
    expected.square().mean().backward()
    run.recurrence = recurrence
    actual = other(gpu_tokens)
    actual.square().mean().backward()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    for (name,p), (_,q) in zip(model.named_parameters(), other.named_parameters()):
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(q.grad, p.grad, atol=2e-4, rtol=2e-3,
                                       msg=lambda msg: name + ': ' + msg)
    print(json.dumps(dict(status='PASS', checks=[
        'four KS2 FFNs; fixed 96D attention/embedding/head', 'all-input coverage and distinct codes',
        'paired geometry, attention and main-mix initialization', 'exact initial logits',
        'side bypasses attention directly', 'side influences logits through FFNs',
        'gradients cross side links', 'final side discarded', 'GPU forward/all gradients match reference'],
        parameters=rows), indent=2), flush=True)


if __name__ == '__main__':
    main()
