"""Matched learned-projection controls and optional exact-equation GPU geometry."""
import torch
from torch import nn
import study_models as v1
from study_models import ExperimentalCell,FeatureCell,parameter_breakdown,make_optimizer
from side_stream import BaseModel
from experiments.common.ks2_stabilized import KS2StabilizedCell


def build_model(seed,cfg):
    family=cfg.get('family','history')
    if family=='history':
        model=v1.build_model(seed,cfg)
    elif family in ('dense_ks2','mlp'):
        width=cfg.get('width',96)
        model=BaseModel(seed,'mlp',width=width,layers=4,fused_cpu=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+2000)
            for block in model.blocks:
                if family=='mlp':
                    if 'hidden' in cfg:
                        block.key=nn.Linear(width,cfg['hidden'],bias=False)
                        block.value=nn.Linear(cfg['hidden'],width,bias=False)
                        nn.init.uniform_(block.key.weight,-.5/width**.5,.5/width**.5)
                        nn.init.zeros_(block.value.weight)
                else:
                    cell=KS2StabilizedCell(width,width,num_subspaces=cfg.get('charts',31),geo_steps=1)
                    nn.init.zeros_(cell.up.weight)
                    kind=cfg.get('cell','ks2')
                    if kind in ('identity','relu2','gelu'):cell=FeatureCell(cell,kind)
                    elif any(k in cfg for k in ('cell','coord_scale','coord_skip','coord_norm')):cell=ExperimentalCell(cell,cfg)
                    if cfg.get('input_init')=='mlp':nn.init.uniform_(cell.down.weight,-.5/width**.5,.5/width**.5)
                    block.ffn=cell
                    del block.key,block.value
        # Dense up matrices use the MLP learning rate unless explicitly changed.
        cfg=dict(cfg)
        cfg.setdefault('output_lr',1.)
    else:raise ValueError(family)
    model.study_config=cfg
    if cfg.get('fused_geometry'):
        from ks2_geometry_gpu import install
        for block in model.blocks:
            if hasattr(block,'ffn') and isinstance(block.ffn,KS2StabilizedCell):install(block.ffn)
    return model
