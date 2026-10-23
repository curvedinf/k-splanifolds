"""Compatibility import; the implementation is now export_ks3."""
import sys
if __package__:
    from . import export_ks3 as _implementation
else:
    import export_ks3 as _implementation
if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
