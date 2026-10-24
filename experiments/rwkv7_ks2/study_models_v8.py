"""Fixed coordinate shuffle between shared KS2 geometry iterations.

The permutation is undone after the odd iteration, preserving the direct
coordinate residual when geometry is zero. No learned parameters are added.
"""
import types
from torch.nn import functional as F
from study_models_v7 import build_model as build_v7,make_optimizer,parameter_breakdown
from experiments.common.ks2_stabilized import _coordinates


def shuffled_forward(self,x):
    coords=self.down(x).float().reshape(-1,self.num_subspaces,self.subspace_rank)
    for i in range(self.study_geometry_steps):
        coords=_coordinates(coords)
        if i%2:
            mixed=coords.transpose(1,2).contiguous().reshape_as(coords)
            mixed=mixed+self.study_geometry_step_scale*self.geo_step(mixed)
            coords=mixed.reshape(-1,self.subspace_rank,self.num_subspaces).transpose(1,2).contiguous()
        else:
            coords=coords+self.study_geometry_step_scale*self.geo_step(coords)
    if self.study_outer_kind=='ks2_relu2':coords=F.relu(coords).square()
    return self.up(coords.reshape(*x.shape[:-1],-1))


def build_model(seed,cfg):
    model=build_v7(seed,cfg)
    if cfg.get('geometry_shuffle'):
        assert cfg.get('geometry_steps',0)>=2
        for b in model.blocks:b.ffn.forward=types.MethodType(shuffled_forward,b.ffn)
    return model
