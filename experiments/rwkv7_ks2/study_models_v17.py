"""Scale only spine geometry inputs; preserve projections and raw bypass."""
import types
from study_models_v15 import build_model as build_v15, make_optimizer, parameter_breakdown


def scaled_spine(self, coordinates):
    return self.study_unscaled_spine(coordinates * self.study_spine_input_scale)


def build_model(seed, cfg):
    scale = cfg.get('spine_input_scale')
    if scale is not None:
        assert cfg.get('family') == 'dense_ks2' and cfg.get('geometry_terms') == 'spine'
        assert float(scale) > 0
        assert not any(cfg.get(key) for key in (
            'fused_geometry', 'fused_partial', 'fused_general', 'geometry_input_norm',
            'geometry_steps', 'geometry_placement', 'geometry_equation', 'geometry_input_sigmoid'))
    model = build_v15(seed, cfg)
    if scale is not None:
        for block in model.blocks:
            cell = block.ffn
            cell.study_spine_input_scale = float(scale)
            cell.study_unscaled_spine = cell.geo_step
            cell.geo_step = types.MethodType(scaled_spine, cell)
    return model
