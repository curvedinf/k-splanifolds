"""Run the established FP32 GPU protocol with a KS2-only FFN side stream."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import run
import run_spread_gpu
from side_stream import SideModel, parameter_breakdown


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--side-width', type=int, required=True)
    options, remaining = parser.parse_known_args()
    out = Path(remaining[remaining.index('--out') + 1])

    def model_factory(seed, variant, **kwargs):
        assert kwargs.get('width', 96) == 96 and kwargs.get('layers', 4) == 4
        assert kwargs.get('ffn_charts', 32) == 32
        return SideModel(seed, options.side_width)

    optimizer_factory = run.make_optimizer
    run.Model = model_factory
    run.make_optimizer = lambda model, control_update_scale=False, ffn_charts=32: optimizer_factory(
        model, control_update_scale, 2)  # fixed decoder gains retain 5.5x LR
    # SideModel already installs the fixed projections on all four FFNs.
    run_spread_gpu.install = lambda model, coding: model
    sys.argv = [sys.argv[0], '--coding', 'fixed_both', *remaining]
    run_spread_gpu.main()
    path = out / 'results.json'
    if not path.exists():
        return
    result = json.loads(path.read_text())
    result['arguments']['side_width'] = options.side_width
    result['architecture'] = 'four KS2 FFNs; 96D attention/embedding/head; FFN-only side residual'
    result['output_lr_multiplier'] = 5.5
    total_width = 96 + options.side_width
    result['coding_protocol'].update(
        width=total_width, charts=32,
        input_coverage=f'all {total_width} FFN inputs in every coordinate of all 32 subspaces',
        output=f'fixed transpose with unit row norms and {total_width} zero-init learned gains per FFN',
        note='32 independent fixed codes; main width 96; zero initial side stream bypasses attention; '
             'combined FFN LayerNorm/time mix; final side outputs discarded; all depth gradients retained.')
    result['parameter_breakdown'] = parameter_breakdown(SideModel(result['arguments']['seed'], options.side_width))
    for name in ('side_stream.py', 'run_side_gpu.py'):
        result['source_hashes'][name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
