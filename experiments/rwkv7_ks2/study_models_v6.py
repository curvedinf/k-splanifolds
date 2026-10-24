"""Arbitrary-width MLP controls with explicitly matched input-weight scale."""
import torch
from study_models_v5 import build_model as build_v5,make_optimizer,parameter_breakdown
from study_models_v4 import ActivatedReadout

def build_model(seed,cfg):
    requested=cfg.get('cell')
    feature=requested in ('square','relu','silu')
    model=build_v5(seed,dict(cfg,cell='identity') if feature else cfg)
    if feature:
        for b in model.blocks:
            if hasattr(b,'ffn'):b.ffn.up=ActivatedReadout(b.ffn.up,requested)
        model.study_config['cell']=requested
    if cfg.get('mlp_input_scale',1.)!=1.:
        assert cfg.get('family')=='mlp'
        with torch.no_grad():
            for b in model.blocks:b.key.weight.mul_(cfg['mlp_input_scale'])
    return model
