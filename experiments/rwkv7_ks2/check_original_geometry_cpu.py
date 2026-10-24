"""Original geometry reference, fixed projections, and extended-factory checks."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path

import torch
from torch.nn import functional as F
import run
from experiments.common.ks2 import KS2Cell
from study_models_v14 import build_model as build_v14
from study_models_v15 import build_model, make_optimizer, parameter_breakdown


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(1923)
    source = a.root / 'prior_additional/rhac_source/ks2.py'
    spec = importlib.util.spec_from_file_location('rhac_reference_ks2', source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    geometry = []
    for rank in (1, 8):
        cell = KS2Cell(5, 5, num_subspaces=3, subspace_rank=rank, geo_steps=1)
        ref = reference.KS2BlockBank(3, rank, geo_steps=1, dtype=torch.float32)
        with torch.no_grad():
            for name, param in cell.named_parameters(recurse=False):
                getattr(ref, name).copy_(param)
        for scenario in ('positive', 'signed', 'zero_sum'):
            x = torch.randn(17, 3, rank)
            if scenario == 'positive':
                x = torch.sigmoid(x / 1.5)
            elif scenario == 'zero_sum':
                x = x - x.mean(-1, keepdim=True)
            x.requires_grad_()
            y = x.detach().clone().requires_grad_()
            expected = ref._eval_once(x)
            actual = cell.geo_step(y)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            upstream = torch.randn_like(actual)
            names = [n for n, _ in cell.named_parameters(recurse=False)]
            g = torch.autograd.grad(expected, (x, *(getattr(ref, n) for n in names)), grad_outputs=upstream)
            h = torch.autograd.grad(actual, (y, *(getattr(cell, n) for n in names)), grad_outputs=upstream)
            for expected_grad, actual_grad in zip(g, h):
                torch.testing.assert_close(actual_grad, expected_grad, atol=2e-5, rtol=2e-4)
            geometry.append(dict(rank=rank, scenario=scenario, forward_max_error=(actual-expected).abs().max().item(), status='PASS'))
    manifest = json.loads((a.root / 'wave12c_original_geometry_manifest.json').read_text())
    configs = {j['config']['name']: j['config'] for j in manifest['jobs']}
    tokens = torch.randint(256, (2, 9))
    rows = []
    for seed in (11, 22, 33):
        control = run.Model(seed, 'mlp', width=96, layers=4, fused_cpu=False)
        wrapped = build_model(seed, dict(name='original_mlp', family='mlp'))
        assert control.state_dict().keys() == wrapped.state_dict().keys()
        assert all(torch.equal(v, wrapped.state_dict()[n]) for n, v in control.state_dict().items())
        original_opt, wrapped_opt = run.make_optimizer(control), make_optimizer(wrapped)
        assert len(original_opt.param_groups) == len(wrapped_opt.param_groups)
        for x, y in zip(original_opt.param_groups, wrapped_opt.param_groups):
            assert {k:v for k,v in x.items() if k != 'params'} == {k:v for k,v in y.items() if k != 'params'}
            assert len(x['params']) == len(y['params'])
            assert all(torch.equal(p, q) for p, q in zip(x['params'], y['params']))
    for name, cfg in configs.items():
        model = build_model(11, cfg)
        base = build_v14(11, {k:v for k,v in cfg.items() if k not in ('geometry_equation', 'geometry_input_sigmoid', 'geometry_placement')})
        assert model.state_dict().keys() == base.state_dict().keys()
        assert all(torch.equal(v, base.state_dict()[n]) for n,v in model.state_dict().items())
        with torch.no_grad():
            torch.testing.assert_close(model(tokens), base(tokens), atol=0, rtol=0)
            for b in model.blocks:
                b.ffn.up.weight.normal_(std=.01)
        logits = model(tokens)
        loss = F.cross_entropy(logits[:, :-1].flatten(0,1), tokens[:, 1:].flatten())
        loss.backward()
        for n, p in model.named_parameters():
            if n == 'blocks.0.attention.value_bias':
                assert p.grad is None
            else:
                assert p.grad is not None and torch.isfinite(p.grad).all(), (name, n)
        changed = tokens.clone()
        changed[:, 5:] = (changed[:, 5:] + 1) % 256
        with torch.no_grad():
            torch.testing.assert_close(model(tokens)[:, :5], model(changed)[:, :5], atol=1e-7, rtol=1e-7)
        count = parameter_breakdown(model)['total']
        assert count == 613168
        rows.append(dict(name=name, parameters=count, status='PASS'))
    result = dict(status='PASS', original_equation_reference=geometry, model_cases=rows,
                  original_mlp_state_and_optimizer_seeds=[11,22,33], note='CPU checks. GPU model/gradient/causality validation remains mandatory before training.')
    (a.root / 'original_geometry_cpu_check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
