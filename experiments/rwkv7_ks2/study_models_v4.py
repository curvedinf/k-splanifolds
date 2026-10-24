"""Activation-shape and frozen-geometry controls on the validated study models."""
from torch import nn
from torch.nn import functional as F
from study_models_v3 import build_model as build_v3,make_optimizer,parameter_breakdown

class ActivatedReadout(nn.Module):
    def __init__(self,base,kind):super().__init__();self.inner=base;self.kind=kind
    @property
    def weight(self):return self.inner.weight
    def forward(self,x):
        if self.kind=='relu':x=F.relu(x)
        elif self.kind=='square':x=x.square()
        elif self.kind=='silu':x=F.silu(x)
        else:raise ValueError(self.kind)
        return self.inner(x)

def build_model(seed,cfg):
    model=build_v3(seed,cfg)
    for b in model.blocks:
        if not hasattr(b,'ffn'):continue
        cell=b.ffn
        if cfg.get('outer_activation'):
            assert cfg.get('cell','ks2')=='ks2' and cfg.get('coord_scale',1.)==1.
            cell.up=ActivatedReadout(cell.up,cfg['outer_activation'])
        if cfg.get('fused_partial'):
            from ks2_partial_gpu import install_partial
            install_partial(cell)
        if cfg.get('frozen_geometry'):
            for name,p in list(cell.named_parameters(recurse=False)):
                if name.startswith('geo_'):
                    delattr(cell,name);cell.register_buffer(name,p.detach().clone())
    return model
