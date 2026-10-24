"""Compose rank/basis/selective-learning wins with post-freezing calibration."""
import types
import torch
from torch import nn
from study_models_v17 import build_model as build_v17, make_optimizer, parameter_breakdown
from study_models_v12 import calibrated_geo


def build_model(seed, cfg):
    mode = cfg.get('geometry_calibration')
    if not mode:
        return build_v17(seed, cfg)
    assert cfg.get('family') == 'dense_ks2' and cfg.get('geometry_terms') == 'spine'
    assert cfg.get('frozen_geometry') or cfg.get('geometry_trainable') == 'warps'
    assert mode in ('bias', 'gain', 'bias_gain')
    base = {key: value for key, value in cfg.items() if key != 'geometry_calibration'}
    model = build_v17(seed, base)
    # Calibration must be added after generalized-rank freezing and fusion.
    # Otherwise the old factory either rejects the composition or freezes it.
    for block in model.blocks:
        cell = block.ffn
        shape = (1, cell.num_subspaces, cell.subspace_rank)
        if mode in ('bias', 'bias_gain'):
            cell.geo_feature_bias = nn.Parameter(torch.zeros(shape))
        if mode in ('gain', 'bias_gain'):
            cell.geo_feature_gain = nn.Parameter(torch.ones(shape))
        cell.study_uncalibrated_geo = cell.geo_step
        cell.geo_step = types.MethodType(calibrated_geo, cell)
    model.study_config['geometry_calibration'] = mode
    return model
