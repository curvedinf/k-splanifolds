"""Build both kernels with explicit ISA flags; no unsafe fast-math."""

import os
import shlex
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).parent
    output = root / '.build'
    output.mkdir(exist_ok=True)
    common = ['-O3', '-std=c++17', '-fPIC', '-shared', '-fopenmp', '-ffp-contract=fast',
              '-mavx2', '-mfma', '-mavx512f', '-mavx512bw', '-mavx512dq', '-mavx512vl']
    for name in ('horner', 'vnni'):
        flags = common + (['-mavx512vnni'] if name == 'vnni' else [])
        subprocess.run(shlex.split(os.environ.get('CXX', 'g++')) + flags +
                       [str(root / (name + '.cpp')), '-o', str(output / (name + '.so'))], check=True)
        print(output / (name + '.so'))


if __name__ == '__main__':
    main()
