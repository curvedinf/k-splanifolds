#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Core deps for experiments (CPU-only torch)
python -m pip install numpy scipy
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
