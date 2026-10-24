"""Isolate the original KS2 equation and RHAC-style bounded geometry inputs."""
import types
import torch
from experiments.common.ks2 import KS2Cell
from study_models_v14 import build_model as build_v14, make_optimizer, parameter_breakdown


def bounded_geometry(self, coords):
    return self.study_unbounded_geo(torch.sigmoid(coords / 1.5))


def build_model(seed, cfg):
    equation = cfg.get('geometry_equation')
    bounded = cfg.get('geometry_input_sigmoid', False)
    if equation or bounded:
        assert cfg.get('family') == 'dense_ks2'
        assert not cfg.get('geometry_terms')
        assert not any(cfg.get(k) for k in ('fused_geometry', 'fused_partial', 'fused_general',
                                           'geometry_input_norm', 'geometry_basis', 'geometry_steps'))
        assert equation in (None, 'original')
    model = build_v14(seed, cfg)
    if equation or bounded:
        for block in model.blocks:
            cell = block.ffn
            if equation == 'original':
                cell.geo_step = types.MethodType(KS2Cell.geo_step, cell)
            if bounded:
                cell.study_unbounded_geo = cell.geo_step
                cell.geo_step = types.MethodType(bounded_geometry, cell)
    return model
