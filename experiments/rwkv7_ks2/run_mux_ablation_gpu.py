"""Established GPU protocol with one independently selected mux ablation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import run
import run_spread_gpu
from mux_ablations import MODES, DEFAULT_CODE_SEED, build_model, make_optimizer, parameter_breakdown


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mode',choices=MODES,required=True)
    parser.add_argument('--code-seed',type=int,default=DEFAULT_CODE_SEED)
    opts, remaining = parser.parse_known_args()
    out = Path(remaining[remaining.index('--out')+1])
    def factory(seed,variant,**kwargs):
        assert kwargs.get('width',96)==96 and kwargs.get('layers',4)==4
        return build_model(seed,kwargs.get('ffn_charts',64),opts.mode,opts.code_seed)
    run.Model, run.make_optimizer = factory, make_optimizer
    run_spread_gpu.install = lambda model,coding:model
    sys.argv = [sys.argv[0],'--coding','fixed_both',*remaining]
    run_spread_gpu.main()
    path = out/'results.json'
    if not path.exists(): return
    result = json.loads(path.read_text())
    result['arguments'].update(mode=opts.mode,code_seed=opts.code_seed,attention_skip=False)
    result['architecture'] = 'Independent mux hypothesis: '+opts.mode
    result['output_lr_multiplier'] = 5.5
    result['coding_protocol'] = dict(mode=opts.mode,view_seed=opts.code_seed,charts=64,rank=8,
        attention_skip=False,description={
        'baseline':'Unchanged raw-mux, no attention skip, final-only demux.',
        'head_skip':'Only add last post-attention 96D state to final demux before head norm.',
        'output_gains':'Only add per-layer 512D gains on raw outputs, init .05, LR 5.5x and inverse WD; no head bypass.',
        'persistent_state':'Only accumulate raw updates in a persistent 512D state; history stores successive state snapshots; final demux reads final state.',
        'source_norm':'Only normalize each embedding/attention/raw source independently with nonaffine LayerNorm before attention mux and FFN global norm/mux; history storage remains raw.',
        'demuxed_history':'Original 64-chart/2048-side separate post-attention/post-FFN history reference; default seed exactly reproduces original constructor.'}[opts.mode])
    result['parameter_breakdown'] = parameter_breakdown(build_model(result['arguments']['seed'],64,opts.mode,opts.code_seed))
    for name in ('side_stream.py','side_history.py','raw_mux.py','mux_ablations.py','run_mux_ablation_gpu.py'):
        result['source_hashes'][name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__': main()
