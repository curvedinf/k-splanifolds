"""Mixed-rank spine charts and an optional parallel MLP feature branch."""
import torch
from torch import nn
from torch.nn import functional as F
from study_models_v18 import build_model as build_v18, make_optimizer, parameter_breakdown
from study_models_v3 import TermCell
from experiments.common.ks2_stabilized import KS2StabilizedCell, _coordinates


class MixedSpineCell(nn.Module):
    def __init__(self, base, groups, mlp_features, seed, fused):
        super().__init__()
        self.down, self.up = base.down, base.up
        self.mixed_ranks = groups
        self.mlp_features = mlp_features
        self.geometry = nn.ModuleList()
        self.feature_slices = []
        self.mlp_slice = None
        coefficients = base.geo_spine.detach().permute(0, 2, 1).reshape(-1, 6)
        offset = 0
        with torch.random.fork_rng(devices=[]):
            for index, (rank, charts) in enumerate(groups):
                assert rank in (1, 2, 4, 8, 16) and charts > 0
                torch.manual_seed(seed + index)
                original = KS2StabilizedCell(self.down.in_features, self.up.out_features, charts, rank, 1)
                cell = TermCell(original, dict(cell='ks2_relu2', geometry_terms='spine'))
                del cell.down, cell.up
                width = rank * charts
                with torch.no_grad():
                    cell.geo_spine.copy_(coefficients[offset:offset + width].reshape(charts, rank, 6).permute(0, 2, 1))
                if fused:
                    from ks2_general_gpu import install
                    install(cell)
                self.geometry.append(cell)
                self.feature_slices.append((offset, offset + width))
                offset += width
                if index == 0 and mlp_features:
                    self.mlp_slice = (offset, offset + mlp_features)
                    offset += mlp_features
        assert offset == self.down.out_features == self.up.in_features

    def forward(self, x):
        coordinates = _coordinates(self.down(x))
        outputs = []
        offset = 0
        for index, ((rank, charts), cell) in enumerate(zip(self.mixed_ranks, self.geometry)):
            width = rank * charts
            local = coordinates[..., offset:offset + width].contiguous().reshape(-1, charts, rank)
            outputs.append(F.relu(local + cell.geo_step(local)).square().reshape(*x.shape[:-1], width))
            offset += width
            if index == 0 and self.mlp_features:
                outputs.append(F.relu(coordinates[..., offset:offset + self.mlp_features]).square())
                offset += self.mlp_features
        return self.up(torch.cat(outputs, dim=-1))


def build_model(seed, cfg):
    if 'mixed_ranks' not in cfg:
        return build_v18(seed, cfg)
    assert cfg['family'] == 'dense_ks2' and cfg['cell'] == 'ks2_relu2'
    assert cfg['geometry_terms'] == 'spine'
    assert not any(cfg.get(key) for key in ('frozen_geometry', 'geometry_calibration', 'geometry_basis', 'fused_geometry', 'fused_partial'))
    groups = [tuple(pair) for pair in cfg['mixed_ranks']]
    mlp_features = cfg.get('mlp_features', 0)
    features = sum(rank * charts for rank, charts in groups) + mlp_features
    assert features == 368 and mlp_features >= 0
    base = {key: value for key, value in cfg.items() if key not in ('mixed_ranks', 'mlp_features', 'fused_general')}
    base['charts'] = features // 8
    model = build_v18(seed, base)
    for index, block in enumerate(model.blocks):
        block.ffn = MixedSpineCell(block.ffn, groups, mlp_features, seed + 44000 + 100 * index, cfg.get('fused_general', False))
    model.study_config = dict(cfg, output_lr=cfg.get('output_lr', 1.))
    return model
