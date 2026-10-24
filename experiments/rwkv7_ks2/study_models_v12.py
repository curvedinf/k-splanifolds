"""Minimal adaptation of fixed KS2 features: warp, bias, and geometry gain."""
import types
import torch
from torch import nn
from study_models_v11 import build_model as build_v11,make_optimizer,parameter_breakdown


def calibrated_geo(self,x):
    y=self.study_uncalibrated_geo(x)
    if hasattr(self,'geo_feature_gain'):y=y*self.geo_feature_gain
    if hasattr(self,'geo_feature_bias'):y=y+self.geo_feature_bias
    return y


def build_model(seed,cfg):
    model=build_v11(seed,cfg)
    mode=cfg.get('geometry_calibration')
    if mode:
        assert cfg.get('family')=='dense_ks2' and cfg.get('geometry_terms')=='spine'
        assert cfg.get('frozen_geometry') or cfg.get('geometry_trainable')=='warps'
        assert mode in ('bias','gain','bias_gain')
        for block in model.blocks:
            cell=block.ffn;shape=(1,cell.num_subspaces,cell.subspace_rank)
            if mode in ('bias','bias_gain'):cell.geo_feature_bias=nn.Parameter(torch.zeros(shape))
            if mode in ('gain','bias_gain'):cell.geo_feature_gain=nn.Parameter(torch.ones(shape))
            cell.study_uncalibrated_geo=cell.geo_step;cell.geo_step=types.MethodType(calibrated_geo,cell)
    return model
