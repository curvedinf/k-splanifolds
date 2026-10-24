"""Wider residual streams with the same 32-chart rank-8 fixed KS2 coding."""
import hashlib
import json
from pathlib import Path
import sys
import run_spread32_gpu


def main():
    argv = sys.argv[1:]
    out = Path(argv[argv.index('--out') + 1])
    run_spread32_gpu.main()
    path = out / 'results.json'
    if path.exists():
        result = json.loads(path.read_text())
        width = result['arguments']['width']
        result['coding_protocol']['input_coverage'] = f'all {width} inputs in every coordinate of all 32 charts'
        result['source_hashes'][Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
