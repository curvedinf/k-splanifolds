"""Verify the four-layer hybrid changes only the two middle FFNs."""
import copy
import torch
import run
from gpu_recurrence import recurrence

torch.set_num_threads(2)
base = run.Model(11, 'mlp', width=96, layers=4, fused_cpu=False)
hybrid = run.Model(11, 'inner_ks2_ffn', width=96, layers=4, ffn_charts=16, fused_cpu=False)
assert [hasattr(block, 'ffn') for block in hybrid.blocks] == [False, True, True, False]
for block in hybrid.blocks:
    for name in ('decay_control', 'rate_control'):
        assert isinstance(getattr(block.attention, name), run.Controller)
original = base.state_dict()
for name, value in hybrid.state_dict().items():
    if '.ffn.' not in name:
        torch.testing.assert_close(value, original[name], atol=0, rtol=0)
tokens = torch.randint(256, (4, 64))
torch.testing.assert_close(base(tokens), hybrid(tokens), atol=0, rtol=0)
counts = [sum(p.numel() for module in ([block.ffn] if hasattr(block, 'ffn') else
                                     [block.key, block.value]) for p in module.parameters())
          for block in hybrid.blocks]
assert counts == [73728, 37696, 37696, 73728]
optimizer = run.make_optimizer(hybrid, True, 16)
assert {id(p) for p in optimizer.param_groups[1]['params']} == {
    id(hybrid.blocks[i].ffn.up.weight) for i in (1, 2)}
assert len([p for group in optimizer.param_groups for p in group['params']]) == len(list(hybrid.parameters()))
print('PASS: MLP/KS2/KS2/MLP, all original controllers, identical shared initialization and initial logits', flush=True)
print('FFN parameters:', counts, 'KS2/MLP:', counts[1] / counts[0], flush=True)
print('Total parameters:', sum(p.numel() for p in base.parameters()),
      sum(p.numel() for p in hybrid.parameters()), flush=True)

hybrid = hybrid.cuda()
with torch.no_grad():
    for block in hybrid.blocks:
        block.attention.output.weight.normal_(std=.01)
        (block.ffn.up.weight if hasattr(block, 'ffn') else block.value.weight).normal_(std=.01)
other = copy.deepcopy(hybrid)
tokens = tokens.cuda()
expected = hybrid(tokens)
expected.square().mean().backward()
run.recurrence = recurrence
actual = other(tokens)
actual.square().mean().backward()
torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
for (name, p), (_, q) in zip(hybrid.named_parameters(), other.named_parameters()):
    if p.grad is not None:
        torch.testing.assert_close(q.grad, p.grad, atol=2e-4, rtol=2e-3,
                                   msg=lambda msg: name + ': ' + msg)
print('PASS: four-layer/three-head GPU model logits and parameter gradients against reference', flush=True)
