"""Compatibility import; the implementation is now ks3_gpu."""
import sys
if __package__:
    from . import ks3_gpu as _implementation
else:
    import ks3_gpu as _implementation
sys.modules[__name__] = _implementation
