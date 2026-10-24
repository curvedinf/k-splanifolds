"""Select a validated equivalent geometry implementation without changing the model."""
def with_fused_geometry(cfg):
    assert cfg.get('family')=='dense_ks2'
    result={k:v for k,v in cfg.items() if k not in ('fused_geometry','fused_partial','fused_general')}
    # The original equation / sigmoid-input screen has only an eager reference.
    if cfg.get('geometry_equation') or cfg.get('geometry_input_sigmoid'):return result
    generalized=cfg.get('subspace_rank',8)!=8 or cfg.get('geometry_basis')
    reduced_grad=cfg.get('frozen_geometry') or cfg.get('geometry_trainable')
    wrapped=cfg.get('geometry_calibration') or cfg.get('geometry_input_norm') or cfg.get('geometry_placement')
    if generalized or (reduced_grad and not wrapped):
        assert not wrapped
        result['fused_general']=True
    elif cfg.get('geometry_terms') in ('spine','undamped'):result['fused_partial']=True
    elif cfg.get('geometry_terms','full')=='full':result['fused_geometry']=True
    else:raise ValueError('No validated fusion for this geometry')
    return result
