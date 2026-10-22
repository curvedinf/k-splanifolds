"""Check width scaling, parameter ownership, and AdamW shrinkage."""
import torch
from run import Model, make_optimizer, set_learning_rate

torch.set_num_threads(2)
for width in (11, 22, 44, 88, 176):
    model = Model(11, 'ks2_ffn', ffn_charts=width)
    optimizer = make_optimizer(model, True, width)
    identifiers = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(identifiers) == len(set(identifiers)) == len(list(model.parameters()))
    assert {id(p) for p in optimizer.param_groups[1]['params']} == {id(block.ffn.up.weight) for block in model.blocks}
    for step in (1, 500, 1000):
        set_learning_rate(optimizer, step, 1000)
        regular, output = optimizer.param_groups
        assert abs(output['lr'] / regular['lr'] - 11 / width) < 1e-12
        assert abs(regular['lr'] * regular['weight_decay'] - output['lr'] * output['weight_decay']) < 1e-15
print('PASS: every parameter assigned once, only FFN output projections scaled, AdamW shrinkage preserved')
