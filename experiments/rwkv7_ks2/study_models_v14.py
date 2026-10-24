"""Freeze generalized-rank KS2 only after constructing the matched rank model."""
from study_models_v13 import build_model as build_v13,make_optimizer,parameter_breakdown


def build_model(seed,cfg):
    generalized=cfg.get('subspace_rank',8)!=8
    mode=cfg.get('geometry_trainable');frozen=cfg.get('frozen_geometry',False)
    if generalized and (mode or frozen):
        assert not (mode and frozen)
        base={k:v for k,v in cfg.items() if k not in ('geometry_trainable','frozen_geometry')}
        model=build_v13(seed,base)
        for block in model.blocks:
            cell=block.ffn
            for name,param in list(cell.named_parameters(recurse=False)):
                if not name.startswith('geo_'):continue
                keep=False
                if mode:
                    assert mode in ('warps','coefficients')
                    warp='raw_scale' in name or 'raw_shift' in name
                    keep=warp if mode=='warps' else name in ('geo_spine','geo_t_basis_edge','geo_x_basis_edge')
                if not keep:
                    delattr(cell,name);cell.register_buffer(name,param.detach().clone())
        if frozen:model.study_config['frozen_geometry']=True
        if mode:model.study_config['geometry_trainable']=mode
        return model
    return build_v13(seed,cfg)
