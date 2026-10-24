"""Established single-mux GPU protocol with static or dynamically modulated mux gains."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import run_single_mux_gpu
from single_mux_gains import MuxGainModel, make_optimizer, parameter_breakdown


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--gain-placement',choices=['input','output','dynamic'],required=True)
    opts,remaining = parser.parse_known_args()
    out = Path(remaining[remaining.index('--out')+1])
    run_single_mux_gpu.SingleMuxModel = lambda seed,charts=64:MuxGainModel(seed,charts,opts.gain_placement)
    run_single_mux_gpu.run_raw_mux_gpu.make_optimizer = make_optimizer
    run_single_mux_gpu.run_raw_mux_gpu.parameter_breakdown = parameter_breakdown
    sys.argv = [sys.argv[0],*remaining]
    run_single_mux_gpu.main()
    path = out/'results.json'
    if not path.exists():return
    result = json.loads(path.read_text())
    result['architecture'] += '; learned per-feature gains at shared-mux '+opts.gain_placement
    result['arguments']['gain_placement'] = opts.gain_placement
    widths = [608]*4 if opts.gain_placement=='output' else [96,704,1312,1920]
    result['coding_protocol']['mux_gains'] = dict(per_layer=widths,total=sum(widths),initial_value=1.,
        lr_multiplier=5.5,weight_decay=.01/5.5,
        placement=('After the single fixed matrix multiply, before splitting KS2/RWKV coordinates.'
                   if opts.gain_placement=='output' else
                   'After shared input LayerNorm, before the single fixed matrix multiply. Each normalized history input feature has one gain.'))
    if opts.gain_placement=='dynamic':
        result['architecture'] = 'Single layer-entry mux with 64x8 KS2, one 96D RWKV view and one extra rank-8 KS2 gain modulator; final full-residual demux'
        result['coding_protocol']['mux_output_width'] = 616
        result['coding_protocol']['branches'] += ' Dynamic arm appends 8 control rows to that same matrix multiply. These pass through a separate rank-8 KS2 cell; its eight activations weight eight learned 608D vectors. Normalize the resulting gains to unit RMS over all 608 main outputs before multiplying them into the main mux outputs.'
        result['coding_protocol']['mux_gains'] = dict(
            placement='Output of shared mux, before splitting the main KS2 and RWKV branches.',
            formula='z = KS2(extra_rank8_view); g = z @ V; gain = g / sqrt(max(mean(g*g),1e-8)); output = main_mux_output * gain',
            normalization='Per token, across all 608 main output coordinates, unit RMS.',
            extra_subspaces_per_layer=1,extra_rank=8,view_seed=103480,
            vector_shape_per_layer=[8,608],vector_parameters=19456,
            geometry_parameters=3280,vector_initialization='Normal(0, 1/sqrt(8)); independent seed+3000 RNG; dynamic arm changes the initial function.',
            gain_lr_multiplier=5.5,geometry_lr_multiplier=1.,weight_decay=.01/5.5,
            history='Only the original 608 main outputs enter history. Control subspace is local to its layer, with no separate time mix.',
            gradient_initialization='Nonzero random vectors allow immediate gradients to the extra KS2 geometry.')
    for name in ('single_mux_gains.py','run_single_mux_gains_gpu.py'):
        result['source_hashes'][name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
