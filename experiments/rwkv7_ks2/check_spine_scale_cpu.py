"""Check the geometry-only scale intervention and unchanged control state."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch
import run
from study_models_v15 import build_model as previous
from study_models_v17 import build_model, make_optimizer
from experiments.common.ks2_stabilized import _coordinates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    jobs = json.loads((args.root / 'wave14_spine_scale_manifest.json').read_text())['jobs']
    rows = []
    for job in jobs:
        cfg = job['config']
        seed = job['seed']
        model = build_model(seed, cfg)
        unscaled = previous(seed, {k: v for k, v in cfg.items() if k != 'spine_input_scale'})
        assert model.state_dict().keys() == unscaled.state_dict().keys()
        assert all(torch.equal(tensor, unscaled.state_dict()[name]) for name, tensor in model.state_dict().items())
        assert sum(p.numel() for p in model.parameters()) <= 615936
        tokens = torch.randint(256, (2, 8), generator=torch.Generator().manual_seed(711))
        with torch.no_grad():
            torch.testing.assert_close(model(tokens), unscaled(tokens), atol=0, rtol=0)
        cell = model.blocks[0].ffn
        with torch.no_grad():
            cell.up.weight.normal_(0, .02)
        reference = copy.deepcopy(cell)
        x = torch.randn(2, 5, 96, requires_grad=True)
        expected_x = x.detach().clone().requires_grad_()
        actual = cell(x)
        c = _coordinates(reference.down(expected_x).reshape(-1, reference.num_subspaces, reference.subspace_rank))
        geometry_c = _coordinates(c * cfg['spine_input_scale'])
        t = 4 * torch.tanh(geometry_c.mean(-1, keepdim=True) / 4)
        geometry = torch.einsum('nsm,smr->nsr', reference.spine_basis(t), reference.geo_spine.float())
        expected = reference.up(torch.relu(c + geometry).square().reshape(2, 5, -1))
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        upstream = torch.randn_like(actual)
        actual_grads = torch.autograd.grad((actual * upstream).sum(), [x, *cell.parameters()])
        expected_grads = torch.autograd.grad((expected * upstream).sum(), [expected_x, *reference.parameters()])
        for got, wanted in zip(actual_grads, expected_grads):
            torch.testing.assert_close(got, wanted, atol=3e-6, rtol=3e-6)
        with torch.no_grad():
            assert not torch.allclose(cell.geo_step(c), unscaled.blocks[0].ffn.geo_step(c))
        rows.append(dict(job=job['id'], status='PASS', initialized_state_identical=True,
                         nonzero_readout_forward_and_all_gradients=True,
                         raw_coordinate_bypass_unscaled=True,
                         parameters=sum(p.numel() for p in model.parameters())))
    for seed in (11, 22, 33):
        cfg = dict(name='original_mlp', family='mlp')
        old = previous(seed, cfg)
        new = build_model(seed, cfg)
        assert all(torch.equal(value, new.state_dict()[name]) for name, value in old.state_dict().items())
        def groups(model):
            return [{key: value for key, value in group.items() if key != 'params'} for group in make_optimizer(model).param_groups]
        assert groups(old) == groups(new)
    result = dict(status='PASS', rows=rows, original_mlp_all_three_seeds_unchanged=True,
                  source_sha256=hashlib.sha256(Path(__file__).with_name('study_models_v17.py').read_bytes()).hexdigest())
    (args.root / 'spine_scale_cpu_check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', configs_and_seeds=len(rows))))


if __name__ == '__main__':
    main()
