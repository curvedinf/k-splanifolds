"""Established GPU protocol, with explicit study configuration and source snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import run
import run_spread_gpu
from study_models import build_model,make_optimizer,parameter_breakdown


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config',type=Path,required=True)
    opts,remaining=parser.parse_known_args()
    cfg=json.loads(opts.config.read_text())
    out=Path(remaining[remaining.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parent
    snapshot=out/'sources';snapshot.mkdir(exist_ok=True)
    sources={}
    for name in ('study_models.py','study_run_gpu.py','side_stream.py','side_history.py','spread_projection.py'):
        data=(root/name).read_bytes();(snapshot/name).write_bytes(data)
        sources[name]=hashlib.sha256(data).hexdigest()
    def factory(seed,variant,**kwargs):return build_model(seed,cfg)
    run.Model,run.make_optimizer=factory,make_optimizer
    run_spread_gpu.install=lambda model,coding:model
    sys.argv=[sys.argv[0],'--coding','fixed_both',*remaining]
    run_spread_gpu.main()
    path=out/'results.json'
    if not path.exists():return
    r=json.loads(path.read_text())
    r['study_config']=cfg
    r['ffn_types']=[cfg.get('cell','ks2') if cfg.get('family','history')=='history' else 'mlp']*4
    r['parameter_breakdown']=parameter_breakdown(build_model(r['arguments']['seed'],cfg))
    r['architecture']='Timed KS2 study: '+cfg['name']
    r['source_hashes'].update(sources)
    r['coding_protocol']=dict(baseline='demuxed_history_64_rank8_side2048',config=cfg,
        original_attention=True,input_coverage='Unchanged RHAC all-input fixed codes unless explicitly stated.',
        output='Current main, side, and separate previous attention/decoder history remain baseline.',
        training='Same paired data, seed, FP32, AdamW, global norm clipping and 15k cosine schedule; changed optimizer scales are explicit in configuration.')
    path.write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':main()
