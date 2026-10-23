"""Compatibility import for the canonical root-level ks3_int8 implementation."""
import sys
from pathlib import Path

# Also support historical scripts launched from the experiment directory.
_repository = str(Path(__file__).resolve().parents[2])
if _repository not in sys.path:
    sys.path.insert(0, _repository)
import ks3_int8 as _implementation

sys.modules[__name__] = _implementation
