"""Targeted, configuration-driven KS2 bottleneck interventions for the timed study."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from side_stream import BaseModel
from side_history import HistoryModel
from experiments.common.ks2_stabilized import KS2StabilizedCell,_coordinates


class AffineInput(nn.Module):
    def __init__(self,base,affine=False,rotation=False):
        super().__init__()
        self.register_buffer('weight',base.weight.clone())
        if affine:
            self.gain=nn.Parameter(torch.ones(self.weight.shape[0]))
            self.shift=nn.Parameter(torch.zeros(self.weight.shape[0]))
        if rotation:self.rotation=nn.Parameter(torch.eye(96))
    def forward(self,x):
        if hasattr(self,'rotation'):x=torch.cat((F.linear(x[...,:96],self.rotation),x[...,96:]),-1)
        y=F.linear(x,self.weight)
        if hasattr(self,'gain'):y=y*self.gain+self.shift
        return y


class MainReadout(nn.Module):
    def __init__(self,base,kind,rank=16):
        super().__init__()
        self.register_buffer('matrix',base.matrix.clone())
        self.weight=base.weight
        self.kind=kind
        if kind=='learned':self.main_matrix=nn.Parameter(self.matrix[:96].clone())
        elif kind=='lora':
            self.down=nn.Parameter(torch.empty(rank,self.matrix.shape[1]))
            nn.init.orthogonal_(self.down)
            self.up=nn.Parameter(torch.zeros(96,rank))
        elif kind=='rotation':self.rotation=nn.Parameter(torch.eye(96))
        else:raise ValueError(kind)
    def forward(self,x):
        if self.kind=='learned':
            y=torch.cat((F.linear(x,self.main_matrix),F.linear(x,self.matrix[96:])),dim=-1)
        else:
            y=F.linear(x,self.matrix)
            if self.kind=='lora':y=torch.cat((y[...,:96]+F.linear(F.linear(x,self.down),self.up),y[...,96:]),dim=-1)
        y=y*self.weight
        if self.kind=='rotation':y=torch.cat((F.linear(y[...,:96],self.rotation),y[...,96:]),dim=-1)
        return y


class ExperimentalCell(KS2StabilizedCell):
    def __init__(self,base,cfg):
        nn.Module.__init__(self)
        for name,p in base.named_parameters(recurse=False):self.register_parameter(name,p)
        for name,b in base.named_buffers(recurse=False):self.register_buffer(name,b)
        self.down,self.up=base.down,base.up
        self.num_subspaces,self.subspace_rank,self.geo_steps=base.num_subspaces,base.subspace_rank,base.geo_steps
        self.kind=cfg.get('cell','ks2');self.coord_scale=cfg.get('coord_scale',1.)
        self.skip=cfg.get('coord_skip',1.);self.coord_norm=cfg.get('coord_norm',False)
    def forward(self,x):
        shape=x.shape[:-1]
        coords=self.down(x).float().reshape(-1,self.num_subspaces,self.subspace_rank)*self.coord_scale
        coords=_coordinates(coords)
        if self.coord_norm:coords=coords*coords.square().mean(-1,keepdim=True).clamp_min(1e-6).rsqrt()
        result=coords*self.skip+self.geo_step(coords)
        if self.kind=='ks2_relu2':result=F.relu(result).square()
        if self.kind=='ks2_gelu':result=F.gelu(result)
        return self.up((result/self.coord_scale).reshape(*shape,-1))


class FeatureCell(nn.Module):
    def __init__(self,base,kind):
        super().__init__();self.down,self.up=base.down,base.up
        self.num_subspaces,self.subspace_rank=base.num_subspaces,base.subspace_rank
        self.kind=kind
    def forward(self,x):
        y=self.down(x)
        if self.kind=='relu2':y=F.relu(y).square()
        elif self.kind=='gelu':y=F.gelu(y)
        elif self.kind!='identity':raise ValueError(self.kind)
        return self.up(y)


def build_model(seed,cfg):
    family=cfg.get('family','history')
    if family=='mlp':
        model=BaseModel(seed,'mlp',width=96,layers=4,fused_cpu=False)
    elif family=='history':
        model=HistoryModel(seed,cfg.get('side_width',2048),cfg.get('charts',64))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+7000)
            for block in model.blocks:
                cell=block.ffn
                kind=cfg.get('cell','ks2')
                if kind in ('identity','relu2','gelu'):cell=FeatureCell(cell,kind)
                elif any(k in cfg for k in ('cell','coord_scale','coord_skip','coord_norm')):cell=ExperimentalCell(cell,cfg)
                if cfg.get('mux_affine') or cfg.get('input_rotation'):
                    cell.down=AffineInput(cell.down,cfg.get('mux_affine',False),cfg.get('input_rotation',False))
                if cfg.get('readout','fixed')!='fixed':
                    cell.up=MainReadout(cell.up,cfg['readout'],cfg.get('readout_rank',16))
                block.ffn=cell
    else:raise ValueError(family)
    model.study_config=cfg
    return model


def make_optimizer(model,control_update_scale=False,ffn_charts=64):
    cfg=model.study_config
    output=[b.ffn.up.weight for b in model.blocks if hasattr(b,'ffn')]
    output_ids={id(p) for p in output}
    geometry=[p for n,p in model.named_parameters() if '.geo_' in n]
    geo_scale=cfg.get('geometry_lr',1.)
    geo_ids={id(p) for p in geometry} if geo_scale!=1. else set()
    out_scale=cfg.get('output_lr',5.5)
    global_scale=cfg.get('lr',1.)
    groups=[dict(params=[p for p in model.parameters() if id(p) not in output_ids|geo_ids],
                 lr_scale=global_scale,weight_decay=.01,lr=.001*global_scale)]
    if geometry and geo_scale!=1.:groups.append(dict(params=geometry,lr_scale=global_scale*geo_scale,
        weight_decay=.01/geo_scale,lr=.001*global_scale*geo_scale))
    if output:groups.append(dict(params=output,lr_scale=global_scale*out_scale,
        weight_decay=.01/out_scale,lr=.001*global_scale*out_scale))
    return torch.optim.AdamW(groups,lr=.001)


def parameter_breakdown(model):
    groups={'rwkv':0,'embedding_head':0,'geometry':0,'decoder_readout':0,'decoder_input':0,'other':0}
    for name,p in model.named_parameters():
        if '.attention.' in name:key='rwkv'
        elif name.startswith(('embedding.','head.')):key='embedding_head'
        elif '.geo_' in name:key='geometry'
        elif '.ffn.up.' in name:key='decoder_readout'
        elif '.ffn.down.' in name:key='decoder_input'
        else:key='other'
        groups[key]+=p.numel()
    return dict(total=sum(groups.values()),**groups)
