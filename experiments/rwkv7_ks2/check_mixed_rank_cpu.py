"""Validate mixed chart allocation, unchanged control, gradients and routing."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
import run
from study_models_v18 import build_model as previous
from study_models_v19 import build_model, make_optimizer, parameter_breakdown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(591)
    tokens = torch.randint(256, (2, 8))
    rows = []
    for job in json.loads((args.root / 'wave17_mixed_ranks_manifest.json').read_text())['jobs']:
        cfg = dict(job['config'], fused_general=False)
        model = build_model(job['seed'], cfg)
        base_cfg = {key: value for key, value in cfg.items() if key not in ('mixed_ranks', 'mlp_features')}
        base = previous(job['seed'], base_cfg)
        width = sum(rank * charts for rank, charts in cfg['mixed_ranks']) + cfg.get('mlp_features', 0)
        charts = sum(charts for rank, charts in cfg['mixed_ranks'])
        expected = 2 * 96 * width + 6 * (width - cfg.get('mlp_features', 0)) + 2 * charts
        assert expected <= 73728 and width == 368
        assert next(charts for rank, charts in cfg['mixed_ranks'] if rank == 16) == 2
        with torch.no_grad():
            torch.testing.assert_close(model(tokens), base(tokens), atol=0, rtol=0)
        for name, tensor in base.state_dict().items():
            if '.ffn.' not in name:
                assert torch.equal(tensor, model.state_dict()[name]), name
        for block, original in zip(model.blocks, base.blocks):
            assert sum(p.numel() for p in block.ffn.parameters()) == expected
            assert torch.equal(block.ffn.down.weight, original.ffn.down.weight)
            assert torch.equal(block.ffn.up.weight, original.ffn.up.weight)
            coefficients = original.ffn.geo_spine.permute(0, 2, 1).reshape(-1, 6)
            for geometry, (start, end) in zip(block.ffn.geometry, block.ffn.feature_slices):
                assert torch.equal(geometry.geo_spine.permute(0, 2, 1).reshape(-1, 6), coefficients[start:end])
            if cfg.get('mlp_features'):
                assert block.ffn.mlp_slice == (80, 112)
                assert block.ffn.feature_slices[1:] == [(112,208),(208,336),(336,368)]
            with torch.no_grad():
                block.ffn.up.weight.normal_(0, .02)
        F.cross_entropy(model(tokens).flatten(0, 1), tokens.roll(-1, 1).flatten()).backward()
        for name, parameter in model.named_parameters():
            if name == 'blocks.0.attention.value_bias':
                continue
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            if '.ffn.' in name:
                assert parameter.grad.abs().sum() > 0, name
        changed = tokens.clone()
        changed[:, 5:] = (changed[:, 5:] + 1) % 256
        with torch.no_grad():
            torch.testing.assert_close(model(tokens)[:, :5], model(changed)[:, :5], atol=2e-6, rtol=2e-6)
        for group in make_optimizer(model).param_groups:
            assert abs(group['lr'] * group['weight_decay'] - 1e-5) < 1e-12
        rows.append(dict(job=job['id'], status='PASS', learned_per_ffn=expected,
                         parameters=parameter_breakdown(model), shared_initialized_tensors_identical=True,
                         all_ffn_parameter_gradients_finite_nonzero=True, causal=True))
    for seed in (11, 22, 33):
        control = dict(name='original_mlp', family='mlp')
        old, new = previous(seed, control), build_model(seed, control)
        assert all(torch.equal(value, new.state_dict()[name]) for name, value in old.state_dict().items())
        def groups(model):
            return [{key: value for key, value in group.items() if key != 'params'} for group in make_optimizer(model).param_groups]
        assert groups(old) == groups(new)
        # A single rank-1 group must reduce to the already validated rank-1 model.
        cfg = dict(name='rank1_equivalence', family='dense_ks2', cell='ks2_relu2', geometry_terms='spine')
        old = previous(seed, dict(cfg, charts=368, subspace_rank=1))
        new = build_model(seed, dict(cfg, charts=46, mixed_ranks=[[1, 368]]))
        for a, b in zip(old.blocks, new.blocks):
            with torch.no_grad():
                a.ffn.up.weight.normal_(0, .02)
                b.ffn.up.weight.copy_(a.ffn.up.weight)
        with torch.no_grad():
            torch.testing.assert_close(old(tokens), new(tokens), atol=2e-6, rtol=2e-6)
    result = dict(status='PASS', original_mlp_all_three_seeds_unchanged=True,
                  homogeneous_rank1_equivalence_all_seeds=True, rows=rows,
                  source_sha256=hashlib.sha256(Path(__file__).with_name('study_models_v19.py').read_bytes()).hexdigest())
    (args.root / 'mixed_rank_cpu_check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', cases=len(rows), counts={r['job']:r['learned_per_ffn'] for r in rows})))


if __name__ == '__main__':
    main()
