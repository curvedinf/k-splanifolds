"""Optional compact fixed codes; no architecture or training-protocol change."""
from study_models_v10 import build_model as build_v10,make_optimizer,parameter_breakdown


def build_model(seed,cfg):
    model=build_v10(seed,cfg)
    if cfg.get('structured_codes'):
        assert cfg.get('family')=='history' and not cfg.get('prune_final_side')
        from structured_codes import install
        model=install(model)
    return model
