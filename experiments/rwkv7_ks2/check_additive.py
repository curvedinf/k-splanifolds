"""Check additive middle FFNs, paired initialization, optimizer groups and GPU gradients."""
import copy
import torch
from torch.nn import functional as F
import run
from gpu_recurrence import recurrence

torch.set_num_threads(2)
base = run.Model(11, 'mlp', width=96, layers=4, fused_cpu=False)
model = run.Model(11, 'inner_mlp_plus_ks2', width=96, layers=4, ffn_charts=2, fused_cpu=False)
replacement = run.Model(11, 'inner_ks2_ffn', width=96, layers=4, ffn_charts=2, fused_cpu=False)
assert [hasattr(b, 'ffn_extra') for b in model.blocks] == [False, True, True, False]
assert all(hasattr(b, 'key') and hasattr(b, 'value') and not hasattr(b, 'ffn') for b in model.blocks)
for name, value in base.state_dict().items():
    torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)
for i in (1, 2):
    for name, value in model.blocks[i].ffn_extra.state_dict().items():
        torch.testing.assert_close(value, replacement.blocks[i].ffn.state_dict()[name], atol=0, rtol=0)
assert sum(p.numel() for p in model.parameters()) == 625360
assert [sum(p.numel() for p in b.ffn_extra.parameters()) for b in model.blocks if hasattr(b, 'ffn_extra')] == [4712, 4712]
tokens = torch.randint(256, (4, 64))
torch.testing.assert_close(base(tokens), model(tokens), atol=0, rtol=0)
optimizer = run.make_optimizer(model, True, 2)
assert {id(p) for p in optimizer.param_groups[1]['params']} == {id(model.blocks[i].ffn_extra.up.weight) for i in (1, 2)}
assert optimizer.param_groups[1]['lr_scale'] == 5.5
assert optimizer.param_groups[0]['lr_scale'] == 1
all_ids = [id(p) for g in optimizer.param_groups for p in g['params']]
assert len(all_ids) == len(set(all_ids)) == len(list(model.parameters()))
F.cross_entropy(model(tokens).flatten(0, 1), tokens.flatten()).backward()
for i in (1, 2):
    assert model.blocks[i].value.weight.grad.abs().max() > 0
    assert model.blocks[i].ffn_extra.up.weight.grad.abs().max() > 0
print('PASS: full MLPs retained; two 4,712-parameter additive KS2 branches; shared initialization and initial logits exact; both branches receive gradients; only KS2 outputs get scaled LR', flush=True)

model = model.cuda()
model.zero_grad(set_to_none=True)
with torch.no_grad():
    for block in model.blocks:
        block.attention.output.weight.normal_(std=.01)
        block.value.weight.normal_(std=.01)
        if hasattr(block, 'ffn_extra'):
            block.ffn_extra.up.weight.normal_(std=.01)
other = copy.deepcopy(model)
tokens = tokens.cuda()
expected = model(tokens)
expected.square().mean().backward()
run.recurrence = recurrence
actual = other(tokens)
actual.square().mean().backward()
torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
for (name, p), (_, q) in zip(model.named_parameters(), other.named_parameters()):
    if p.grad is not None:
        torch.testing.assert_close(q.grad, p.grad, atol=2e-4, rtol=2e-3,
                                   msg=lambda message: name + ': ' + message)
print('PASS: additive GPU logits and all parameter gradients match reference recurrence', flush=True)
