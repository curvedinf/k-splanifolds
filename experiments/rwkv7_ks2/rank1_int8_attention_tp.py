"""Compatibility import; the implementation is now ks3_attention_tp."""
import sys
if __package__:
    from . import ks3_attention_tp as _implementation
else:
    import ks3_attention_tp as _implementation
sys.modules[__name__] = _implementation
