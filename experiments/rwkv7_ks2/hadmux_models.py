"""Combine KS2 findings inside the original fixed-Hadamard side/history model.

Projection construction stays rank-8. Geometry may regroup the exact same
projected coordinates, so geometry rank does not change the mixing matrices.
"""
import functools
import types
import torch
from torch import nn
import side_stream
import side_history
from side_history import HistoryModel
from study_models import ExperimentalCell, AffineInput, make_optimizer, parameter_breakdown
from study_models_v3 import TermCell
from study_models_v10 import power_basis, convert_coefficients
from study_models_v12 import calibrated_geo
from study_models_v19 import build_model as previous_model
from experiments.common.ks2_stabilized import KS2StabilizedCell


# These functions use local generators. Cache construction only; projection
# modules clone the returned codes and never mutate the cache.
_codes = functools.lru_cache(maxsize=16)(side_stream.make_codes)
side_stream.make_codes = _codes
side_history.make_codes = _codes


def build_model(seed, cfg):
    if cfg.get('family') != 'hadmux':
        return previous_model(seed, cfg)
    features = cfg.get('features', 512)
    rank = cfg.get('geometry_rank', 8)
    assert features % 8 == 0 and features % rank == 0
    assert rank in (1, 2, 4, 8, 16)
    model = HistoryModel(seed, cfg.get('side_width', 2048), features // 8)
    for index, block in enumerate(model.blocks):
        cell = block.ffn
        if rank != 8:
            old = cell
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 33000 + index)
                cell = KS2StabilizedCell(96, 96, features // rank, rank, 1)
            cell.down, cell.up = old.down, old.up
            with torch.no_grad():
                coefficients = old.geo_spine.permute(0, 2, 1).reshape(features // rank, rank, 6).permute(0, 2, 1)
                cell.geo_spine.copy_(coefficients)
        if cfg.get('geometry_terms') == 'spine':
            cell = TermCell(cell, cfg)
        elif cfg.get('cell', 'ks2') != 'ks2':
            cell = ExperimentalCell(cell, cfg)
        if cfg.get('geometry_basis'):
            assert cfg['geometry_basis'] in ('powers', 'taylor')
            with torch.no_grad():
                for name in ('geo_spine', 'geo_t_basis_edge'):
                    if hasattr(cell, name):
                        getattr(cell, name).copy_(convert_coefficients(getattr(cell, name), cfg['geometry_basis']))
            cell.study_basis = cfg['geometry_basis']
            cell.spine_basis = types.MethodType(power_basis, cell)
        mode = cfg.get('geometry_trainable')
        assert mode in (None, 'warps', 'coefficients')
        if cfg.get('frozen_geometry') or mode:
            for name, param in list(cell.named_parameters(recurse=False)):
                if not name.startswith('geo_'):
                    continue
                warp = 'raw_scale' in name or 'raw_shift' in name
                keep = (mode == 'warps' and warp) or (mode == 'coefficients' and not warp)
                if not keep:
                    delattr(cell, name)
                    cell.register_buffer(name, param.detach().clone())
        if cfg.get('fused_general', False):
            from ks2_general_gpu import install
            install(cell)
        if cfg.get('geometry_calibration'):
            assert cfg['geometry_calibration'] == 'bias'
            cell.geo_feature_bias = nn.Parameter(torch.zeros(1, features // rank, rank))
            cell.study_uncalibrated_geo = cell.geo_step
            cell.geo_step = types.MethodType(calibrated_geo, cell)
        if cfg.get('mux_affine'):
            cell.down = AffineInput(cell.down, affine=True)
        block.ffn = cell
    model.study_config = dict(cfg)
    if cfg.get('prune_final_side'):
        from history_final_prune import prune_final_side
        assert cfg.get('side_width', 2048) > 0
        prune_final_side(model)
    return model
