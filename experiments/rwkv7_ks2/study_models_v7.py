"""Repeated geometry updates with shared parameters, matching the reference cell's iteration idea."""
import types
from torch.nn import functional as F
from study_models_v6 import build_model as build_v6,make_optimizer,parameter_breakdown
from experiments.common.ks2_stabilized import _coordinates


def iterated_forward(self,x):
    coords=self.down(x).float().reshape(-1,self.num_subspaces,self.subspace_rank)
    for _ in range(self.study_geometry_steps):
        coords=_coordinates(coords)
        coords=coords+self.study_geometry_step_scale*self.geo_step(coords)
    if self.study_outer_kind=='ks2_relu2':coords=F.relu(coords).square()
    return self.up(coords.reshape(*x.shape[:-1],-1))


def build_model(seed,cfg):
    model=build_v6(seed,cfg)
    if 'geometry_steps' in cfg:
        assert cfg.get('family')=='dense_ks2'
        assert cfg.get('cell','ks2') in ('ks2','ks2_relu2')
        assert cfg.get('coord_scale',1.)==1. and cfg.get('coord_skip',1.)==1.
        assert not cfg.get('coord_norm') and not cfg.get('outer_activation')
        assert cfg['geometry_steps']>=1
        for b in model.blocks:
            cell=b.ffn
            cell.study_geometry_steps=cfg['geometry_steps']
            cell.study_geometry_step_scale=cfg.get('geometry_step_scale',1.)
            cell.study_outer_kind=cfg.get('cell','ks2')
            cell.forward=types.MethodType(iterated_forward,cell)
    return model
