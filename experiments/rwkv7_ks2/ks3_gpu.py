"""Compatibility import for the canonical experiments.common.ks3_int8 implementation."""
import sys
from pathlib import Path

# Also support historical scripts launched from the experiment directory.
_repository = str(Path(__file__).resolve().parents[2])
if _repository not in sys.path:
    sys.path.insert(0, _repository)
from experiments.common import ks3_int8 as _implementation

sys.modules[__name__] = _implementation
