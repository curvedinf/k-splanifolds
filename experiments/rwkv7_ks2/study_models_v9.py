"""KS2-only selective geometry training: separate basis coefficients from warps."""
from study_models_v8 import build_model as build_v8,make_optimizer,parameter_breakdown


def build_model(seed,cfg):
    model=build_v8(seed,cfg)
    if cfg.get('prune_final_side'):
        assert cfg.get('family')=='history'
        from history_final_prune import prune_final_side
        model=prune_final_side(model)
    mode=cfg.get('geometry_trainable')
    if mode:
        assert mode in ('warps','coefficients')
        assert cfg.get('family')=='dense_ks2' and not cfg.get('frozen_geometry')
        for b in model.blocks:
            cell=b.ffn
            for name,param in list(cell.named_parameters(recurse=False)):
                if not name.startswith('geo_'):continue
                is_warp='raw_scale' in name or 'raw_shift' in name
                keep=is_warp if mode=='warps' else name in ('geo_spine','geo_t_basis_edge','geo_x_basis_edge')
                if not keep:
                    delattr(cell,name)
                    cell.register_buffer(name,param.detach().clone())
    return model
