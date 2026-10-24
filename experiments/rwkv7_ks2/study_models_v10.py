"""KS2-only basis conditioning, chart rank, and geometry placement experiments.

Power/Taylor coefficient conversion preserves the initial mathematical function.
Rank changes preserve the learned projection tensors whenever feature counts match.
"""
import math
import types
import torch
from torch import nn
from torch.nn import functional as F
from study_models_v9 import build_model as build_v9,make_optimizer,parameter_breakdown
from study_models import ExperimentalCell
from study_models_v3 import TermCell
from experiments.common.ks2_stabilized import KS2StabilizedCell,_coordinates


def power_basis(self,t):
    z=self.warp(t,self.geo_raw_scale,self.geo_raw_shift)
    divisors=(1.,1.,2.,6.) if self.study_basis=='taylor' else (1.,1.,1.,1.)
    return torch.cat((torch.ones_like(t),t,t.square()/divisors[2],t.pow(3)/divisors[3],2*z.square()-1,4*z.pow(3)-3*z),-1)


def convert_coefficients(p,kind):
    c=p.detach().clone()
    c[:,0]=p[:,0];c[:,1]=p[:,2]
    c[:,2]=-3*p[:,0]+3*p[:,1]-2*p[:,2]-p[:,3]
    c[:,3]=2*p[:,0]-2*p[:,1]+p[:,2]+p[:,3]
    if kind=='taylor':c[:,2]*=2;c[:,3]*=6
    return c


def normalized_geo(self,x):
    if self.study_geometry_norm=='rms':
        x=x*(x.square().mean(-1,keepdim=True)+1e-6).rsqrt()
    elif self.study_geometry_norm=='layer':
        x=x-x.mean(-1,keepdim=True)
        x=x*(x.square().mean(-1,keepdim=True)+1e-6).rsqrt()
    else:raise ValueError(self.study_geometry_norm)
    return self.study_original_geo(x)


def placed_forward(self,x):
    c=_coordinates(self.down(x).float().reshape(-1,self.num_subspaces,self.subspace_rank))
    h=F.relu(c).square()
    if self.study_placement=='parallel':h=h+self.geo_step(c)
    elif self.study_placement=='after_activation':h=h+self.geo_step(h)
    else:raise ValueError(self.study_placement)
    return self.up(h.reshape(*x.shape[:-1],-1))


def build_model(seed,cfg):
    rank=cfg.get('subspace_rank',8)
    basecfg=dict(cfg)
    if rank!=8:
        assert cfg.get('family')=='dense_ks2'
        assert not cfg.get('fused_geometry') and not cfg.get('fused_partial')
        assert not any(k in cfg for k in ('geometry_steps','geometry_shuffle','geometry_trainable','frozen_geometry'))
        features=rank*cfg['charts'];assert features%8==0
        basecfg['charts']=features//8
    if cfg.get('geometry_basis'):
        assert not cfg.get('fused_geometry') and not cfg.get('fused_partial')
    model=build_v9(seed,basecfg)
    if rank!=8:
        with torch.random.fork_rng(devices=[]):
            for i,b in enumerate(model.blocks):
                torch.manual_seed(seed+33000+i)
                old=b.ffn
                cell=KS2StabilizedCell(cfg.get('width',96),cfg.get('width',96),cfg['charts'],rank,1)
                cell.down,cell.up=old.down,old.up
                # Map identical per-feature spine coefficients to new chart grouping.
                with torch.no_grad():
                    cell.geo_spine.copy_(old.geo_spine.permute(0,2,1).reshape(cfg['charts'],rank,6).permute(0,2,1))
                cell=TermCell(cell,cfg) if 'geometry_terms' in cfg else ExperimentalCell(cell,cfg)
                b.ffn=cell
    for b in model.blocks:
        if not hasattr(b,'ffn'):continue
        cell=b.ffn
        if cfg.get('geometry_basis'):
            kind=cfg['geometry_basis'];assert kind in ('powers','taylor')
            with torch.no_grad():
                for name in ('geo_spine','geo_t_basis_edge'):
                    if hasattr(cell,name):getattr(cell,name).copy_(convert_coefficients(getattr(cell,name),kind))
            cell.study_basis=kind;cell.spine_basis=types.MethodType(power_basis,cell)
        if cfg.get('geometry_input_norm'):
            cell.study_geometry_norm=cfg['geometry_input_norm'];cell.study_original_geo=cell.geo_step
            cell.geo_step=types.MethodType(normalized_geo,cell)
        if cfg.get('geometry_placement'):
            assert cfg.get('cell')=='ks2_relu2'
            assert not any(k in cfg for k in ('geometry_steps','geometry_shuffle','coord_norm','coord_scale','coord_skip'))
            cell.study_placement=cfg['geometry_placement'];cell.forward=types.MethodType(placed_forward,cell)
    model.study_config=dict(cfg)
    if cfg.get('family')=='dense_ks2':model.study_config.setdefault('output_lr',1.)
    return model
