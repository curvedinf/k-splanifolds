"""Optional equivalent fused kernels for arbitrary-rank and cubic-basis KS2."""
from study_models_v12 import build_model as build_v12,make_optimizer,parameter_breakdown


def build_model(seed,cfg):
    base=dict(cfg);fused=base.pop('fused_general',False);model=build_v12(seed,base)
    if fused:
        assert cfg.get('family')=='dense_ks2'
        assert not cfg.get('fused_geometry') and not cfg.get('fused_partial')
        assert not cfg.get('geometry_calibration') and not cfg.get('geometry_input_norm') and not cfg.get('geometry_placement')
        from ks2_general_gpu import install
        for block in model.blocks:install(block.ffn)
        model.study_config['fused_general']=True
    return model
