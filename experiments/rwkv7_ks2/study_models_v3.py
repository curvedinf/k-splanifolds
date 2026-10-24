"""Projection factorization, fixed input allocation, and geometry component tests."""
import torch
from torch import nn
from torch.nn import functional as F
from study_models_v2 import build_model as build_v2,make_optimizer,parameter_breakdown
from study_models import ExperimentalCell
from spread_projection import FixedInput,make_codes
from experiments.common.ks2_stabilized import _coordinates,_predicates,_strength

class DirectMainReadout(nn.Module):
    """Zero dense main output; preserve fixed side output with scalar gains."""
    def __init__(self,base):
        super().__init__();self.register_buffer('matrix',base.matrix[96:].clone())
        self.weight=nn.Parameter(base.weight[96:].detach().clone())
        self.main_matrix=nn.Parameter(torch.zeros(96,base.matrix.shape[1]))
    def forward(self,x):return torch.cat((F.linear(x,self.main_matrix),F.linear(x,self.matrix)*self.weight),-1)

class TermCell(ExperimentalCell):
    def __init__(self,base,cfg):
        super().__init__(base,cfg);self.terms=cfg['geometry_terms']
        if self.terms=='spine':
            for name in ('geo_t_basis_edge','geo_x_basis_edge','geo_den_delta','geo_den_weight','geo_edge_raw_scale','geo_edge_raw_shift'):delattr(self,name)
        elif self.terms=='edge':del self.geo_spine
        elif self.terms=='undamped':del self.geo_den_delta,self.geo_den_weight
        else:raise ValueError(self.terms)
    def geo_step(self,coords):
        x=_coordinates(coords);t=4*torch.tanh(x.mean(-1,keepdim=True)/4);basis=self.spine_basis(t)
        if self.terms=='spine':return torch.einsum('nsm,smr->nsr',basis,self.geo_spine.float())
        et=torch.einsum('nsm,smro->nsro',basis,self.geo_t_basis_edge.float())
        ex=torch.einsum('nsrk,skro->nsro',self.edge_basis(x),self.geo_x_basis_edge.float())
        if self.terms=='undamped':return torch.einsum('nsm,smr->nsr',basis,self.geo_spine.float())+(et*ex).sum(2)
        distance,importance=_predicates(x)
        den=1+_strength(self.geo_den_delta)[None]*distance[...,None]+_strength(self.geo_den_weight)[None]*importance[...,None]
        return (et/den*ex).sum(2)

def build_model(seed,cfg):
    model=build_v2(seed,cfg)
    for b in model.blocks:
        if not hasattr(b,'ffn'):continue
        cell=b.ffn
        if cfg.get('direct_main'):cell.up=DirectMainReadout(cell.up)
        if cfg.get('input_fixed'):
            codes,_,_=make_codes(cfg.get('width',96),cell.num_subspaces,8)
            cell.down=FixedInput(codes)
        if 'geometry_init_scale' in cfg:
            with torch.no_grad():
                for name in ('geo_spine','geo_t_basis_edge','geo_x_basis_edge'):getattr(cell,name).mul_(cfg['geometry_init_scale'])
        if cfg.get('edge_t_constant'):
            with torch.no_grad():
                cell.geo_t_basis_edge.zero_();cell.geo_t_basis_edge[:,0].fill_(1.);cell.geo_t_basis_edge[:,1].fill_(1.)
        if 'geometry_terms' in cfg:
            assert not cfg.get('fused_geometry');cell=TermCell(cell,cfg)
        b.ffn=cell
    return model
