"""Exactly the same Hadamard models, with verified faster code construction."""
import functools
import side_stream
import side_history
from hadmux_models import build_model, make_optimizer, parameter_breakdown
from hadmux_codes import make_codes

_codes = functools.lru_cache(maxsize=16)(make_codes)
side_stream.make_codes = _codes
side_history.make_codes = _codes
