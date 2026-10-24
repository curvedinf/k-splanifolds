"""32-chart follow-up; fixed decoder gain LR stays the same as two-chart run.

Fixed decoder rows always have unit norm, so their gains do not inherit
the dense output matrix's chart-count scaling. Keep the already-tested
5.5x gain LR and inverse weight decay. Dense output controls retain 11/S.
"""
import hashlib
import json
from pathlib import Path
import sys
import run
import run_spread_gpu


def main():
    args = sys.argv[1:]
    coding = args[args.index('--coding') + 1]
    out = Path(args[args.index('--out') + 1])
    if coding == 'fixed_both':
        original_optimizer = run.make_optimizer

        def make_optimizer(model, control_update_scale=False, ffn_charts=32):
            return original_optimizer(model, control_update_scale, 2)

        run.make_optimizer = make_optimizer
    run_spread_gpu.main()
    path = out / 'results.json'
    if path.exists():
        result = json.loads(path.read_text())
        result['coding_protocol']['input_coverage'] = 'all 96 inputs in every coordinate of all 32 charts'
        result['output_lr_multiplier'] = 5.5 if coding == 'fixed_both' else 11 / 32
        result['source_hashes'][Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
