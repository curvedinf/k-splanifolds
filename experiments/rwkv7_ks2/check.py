"""Check recurrent orientation/gradients, causality, and paired initialization."""
import torch
from run import Model, recurrence

torch.set_num_threads(2)
torch.manual_seed(123)
args = [torch.randn(2, 5, 2, 4, dtype=torch.float64, requires_grad=True) for _ in range(6)]
assert torch.autograd.gradcheck(recurrence, tuple(args), fast_mode=True)
state = torch.zeros(2, 2, 4, 4, dtype=torch.float64)
expected = []
for pos in range(5):
    read, decay, key, value, normalized_key, rate = [item[:, pos] for item in args]
    transition = torch.diag_embed(decay) - normalized_key.unsqueeze(-1) * (normalized_key * rate).unsqueeze(-2)
    state = state @ transition + value.unsqueeze(-1) * key.unsqueeze(-2)
    expected.append((state @ read.unsqueeze(-1)).squeeze(-1))
torch.testing.assert_close(recurrence(*args), torch.stack(expected, 1))
base, ks2 = Model(11, 'mlp'), Model(11, 'ks2')
tokens = torch.randint(256, (2, 12))
torch.testing.assert_close(base(tokens), ks2(tokens), atol=0, rtol=0)
base_state = base.state_dict()
for name, value in ks2.state_dict().items():
    if 'decay_control' not in name and 'rate_control' not in name:
        torch.testing.assert_close(value, base_state[name], atol=0, rtol=0)
for model in (base, ks2):
    changed = tokens.clone()
    changed[:, 6:] = torch.randint(256, changed[:, 6:].shape)
    torch.testing.assert_close(model(tokens)[:, :6], model(changed)[:, :6])
    for block in model.blocks:
        for name in ('decay_control', 'rate_control'):
            torch.testing.assert_close(getattr(block.attention, name)(torch.randn(2, 12, 128)), torch.zeros(2, 12, 128))
print('PASS: recurrence matrix reference, gradcheck, causal prefixes, identical shared parameters and initial controls/logits')
combined = Model(11, 'ks2_ffn')
torch.testing.assert_close(ks2(tokens), combined(tokens), atol=0, rtol=0)
for name, value in ks2.state_dict().items():
    if name in combined.state_dict():
        torch.testing.assert_close(value, combined.state_dict()[name], atol=0, rtol=0)
assert sum(p.numel() for block in combined.blocks for p in block.ffn.parameters()) == 63096
assert all(not hasattr(block, 'key') and not hasattr(block, 'value') for block in combined.blocks)
print('PASS: combined KS2 preserves initial logits/shared parameters; old FFNs removed')
