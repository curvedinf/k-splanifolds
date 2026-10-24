"""Check FP32 GPU recurrence and complete model derivatives against PyTorch."""
import copy
import torch
from torch.nn import functional as F
import run
from gpu_recurrence import recurrence

torch.set_num_threads(2)
torch.manual_seed(77)
reference = run.recurrence
for length in (1, 7, 64):
    inputs = [torch.randn(4, length, 4, 32) for _ in range(6)]
    inputs[0] *= .1
    inputs[1] = inputs[1].sigmoid() * .2 + .79
    inputs[2] *= .1
    inputs[3] *= .1
    inputs[4] = F.normalize(inputs[4], dim=-1)
    inputs[5] = inputs[5].sigmoid()
    a = [x.cuda().requires_grad_() for x in inputs]
    b = [x.detach().clone().requires_grad_() for x in a]
    expected, actual = reference(*a), recurrence(*b)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    grad = torch.randn_like(expected)
    (expected * grad).sum().backward()
    (actual * grad).sum().backward()
    for x, y in zip(a, b):
        torch.testing.assert_close(y.grad, x.grad, atol=3e-6, rtol=1e-4)
print('PASS: recurrence forward and all six gradients at lengths 1, 7, 64', flush=True)

for variant in ('mlp', 'ks2', 'ks2_ffn'):
    model = run.Model(11, variant, fused_cpu=False).cuda()
    # Nonzero residual outputs exercise the otherwise initially gated gradients.
    with torch.no_grad():
        for block in model.blocks:
            block.attention.output.weight.normal_(std=.01)
            if hasattr(block, 'ffn'):
                block.ffn.up.weight.normal_(std=.01)
            else:
                block.value.weight.normal_(std=.01)
    other = copy.deepcopy(model)
    tokens = torch.randint(256, (4, 64), device='cuda')
    run.recurrence = reference
    expected = model(tokens)
    expected.square().mean().backward()
    run.recurrence = recurrence
    actual = other(tokens)
    actual.square().mean().backward()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    for (name, p), (_, q) in zip(model.named_parameters(), other.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(q.grad, p.grad, atol=2e-4, rtol=2e-3,
                                       msg=lambda msg: name + ': ' + msg)
    print('PASS: complete model logits and parameter gradients:', variant, flush=True)
