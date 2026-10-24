"""Check folded RHAC fidelity, all-input coverage, gradients and paired init."""
import ast
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import run
from spread_projection import make_codes, FixedInput, FixedOutput, install


def main():
    torch.set_num_threads(2)
    reference = Path(__file__).resolve().parents[3] / 'SDGraft/RHAC/src/rhac/rhac.py'
    tree = ast.parse(reference.read_text())
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                   and node.name in ('hadamard', 'SpreadViews')]
    namespace = dict(torch=torch, nn=nn, math=math)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(reference), 'exec'), namespace)
    ref = namespace['SpreadViews'](96, 2, 8, 4242 + 99173)
    codes, perms, signs = make_codes()
    assert torch.equal(perms, ref.perms) and torch.equal(signs, ref.signs)
    assert (codes != 0).all()  # stronger than mere coverage of each 8D set
    for code in codes:
        torch.testing.assert_close(code @ code.T, torch.eye(8), atol=2e-6, rtol=2e-6)
    x = torch.randn(2, 2, 96, requires_grad=True)
    expected = ref.forward_flat(x).reshape(2, 2, 2, 12, 8).sum(-2).flatten(-2) / math.sqrt(12)
    actual = FixedInput(codes)(x)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    probe = torch.randn_like(actual)
    grad = torch.autograd.grad((actual * probe).sum(), x)[0]
    refgrad = torch.autograd.grad((expected * probe).sum(), x)[0]
    torch.testing.assert_close(grad, refgrad, atol=2e-6, rtol=2e-6)
    decoder = FixedOutput(codes)
    assert torch.count_nonzero(decoder(actual)) == 0
    decoder.weight.data.fill_(1.)
    torch.testing.assert_close(decoder(actual), actual @ codes.flatten(0, 1) * math.sqrt(6),
                               atol=2e-6, rtol=2e-6)
    tokens = torch.randint(256, (2, 16))
    base = run.Model(11, 'mlp', width=96, layers=4, fused_cpu=False)
    original = run.Model(11, 'inner_ks2_ffn', width=96, layers=4, ffn_charts=2, fused_cpu=False)
    diagnostics = {}
    for coding in ('learned', 'fixed_input', 'fixed_both'):
        model = install(run.Model(11, 'inner_ks2_ffn', width=96, layers=4,
                                  ffn_charts=2, fused_cpu=False), coding)
        for name, value in model.state_dict().items():
            if '.ffn.' not in name:
                assert torch.equal(value, original.state_dict()[name])
            elif '.geo_' in name:
                assert torch.equal(value, original.state_dict()[name])
        torch.testing.assert_close(model(tokens), base(tokens), atol=0, rtol=0)
        optimizer = run.make_optimizer(model, True, 2)
        assert len({id(p) for g in optimizer.param_groups for p in g['params']}) == len(list(model.parameters()))
        captures = []
        handle = model.blocks[1].ffn.register_forward_pre_hook(lambda m, args: captures.append(args[0].detach()))
        loss = F.cross_entropy(model(tokens).flatten(0, 1), tokens.flatten())
        handle.remove()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        with torch.no_grad():
            first_rms = model.blocks[1].ffn(captures[0]).square().mean().sqrt().item()
        optimizer.zero_grad()
        F.cross_entropy(model(tokens).flatten(0, 1), tokens.flatten()).backward()
        assert model.blocks[1].ffn.geo_spine.grad.abs().max() > 0
        assert model.blocks[1].ffn.up.weight.grad.abs().max() > 0
        diagnostics[coding] = dict(parameters=sum(p.numel() for p in model.parameters()),
                                   coordinates=16, first_update_rms=first_rms)
    print(json.dumps(dict(status='PASS', coverage='96 inputs into each of 16 coordinates',
                          checks=['RHAC reference forward/input gradients', 'orthonormal rows per chart',
                                  'fixed decoder transpose', 'paired geometry/shared weights',
                                  'exact initial logits', 'optimizer coverage', 'nonzero geometry gradients'],
                          diagnostic=diagnostics), indent=2))


if __name__ == '__main__':
    main()
