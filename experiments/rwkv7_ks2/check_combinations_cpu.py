"""Verify composed KS2 trainability, initialization, gradients and fixed control."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
import run
from study_models_v17 import build_model as previous
from study_models_v18 import build_model, make_optimizer, parameter_breakdown

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.manual_seed(591)
tokens = torch.randint(256, (2, 8))
rows = []
for job in json.loads((a.root / 'wave15_combinations_manifest.json').read_text())['jobs']:
    cfg = dict(job['config'], fused_general=False)
    model = build_model(job['seed'], cfg)
    base_cfg = {k: v for k, v in cfg.items() if k != 'geometry_calibration'}
    base = previous(job['seed'], base_cfg)
    assert parameter_breakdown(model)['total'] <= 615936
    for name, tensor in base.state_dict().items():
        assert torch.equal(tensor, model.state_dict()[name]), (job['id'], name)
    for block, original in zip(model.blocks, base.blocks):
        with torch.no_grad():
            block.ffn.up.weight.normal_(0, .02)
            original.ffn.up.weight.copy_(block.ffn.up.weight)
    torch.testing.assert_close(model(tokens), base(tokens), atol=0, rtol=0)
    loss = F.cross_entropy(model(tokens).flatten(0, 1), tokens.roll(-1, 1).flatten())
    loss.backward()
    for name, parameter in model.named_parameters():
        if name == 'blocks.0.attention.value_bias':
            continue
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), (job['id'], name)
        if '.geo_' in name:
            calibration = 'geo_feature_' in name
            if cfg.get('frozen_geometry'):
                assert calibration, name
            if cfg.get('geometry_trainable') == 'warps':
                assert calibration or 'raw_scale' in name or 'raw_shift' in name, name
    altered = tokens.clone()
    altered[:, 5:] = (altered[:, 5:] + 1) % 256
    with torch.no_grad():
        torch.testing.assert_close(model(tokens)[:, :5], model(altered)[:, :5], atol=2e-6, rtol=2e-6)
    # Geometry LR changes learning, while inverse weight decay keeps its
    # multiplicative per-update decay matched to the default group.
    for group in make_optimizer(model).param_groups:
        assert abs(group['lr'] * group['weight_decay'] - 1e-5) < 1e-12
    rows.append(dict(job=job['id'], status='PASS', parameters=parameter_breakdown(model),
                     initialized_base_tensors_identical=True, nonzero_readout_forward_matches=True,
                     intended_parameters_trainable=True, finite_gradients=True, causal=True))
for seed in (11, 22, 33):
    cfg = dict(name='original_mlp', family='mlp')
    old, new = previous(seed, cfg), build_model(seed, cfg)
    assert all(torch.equal(value, new.state_dict()[name]) for name, value in old.state_dict().items())
    def groups(model):
        return [{k: v for k, v in g.items() if k != 'params'} for g in make_optimizer(model).param_groups]
    assert groups(old) == groups(new)
result = dict(status='PASS', original_mlp_all_three_seeds_unchanged=True, rows=rows,
              source_sha256=hashlib.sha256(Path(__file__).with_name('study_models_v18.py').read_bytes()).hexdigest())
(a.root / 'combination_cpu_check.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(status='PASS', configurations_and_seeds=len(rows))))
