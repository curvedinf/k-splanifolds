"""Established 15k GPU protocol for isolated RWKV projection replacements."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import run
import run_spread_gpu
from rwkv_mux import MODES, RWKVMuxModel, make_optimizer, parameter_breakdown


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mode',choices=MODES,required=True)
    opts,remaining=parser.parse_known_args()
    out=Path(remaining[remaining.index('--out')+1])
    def factory(seed,variant,**kwargs):
        assert kwargs.get('width',96)==96 and kwargs.get('layers',4)==4
        return RWKVMuxModel(seed,opts.mode,kwargs.get('ffn_charts',64))
    run.Model,run.make_optimizer=factory,make_optimizer
    run_spread_gpu.install=lambda model,coding:model
    sys.argv=[sys.argv[0],'--coding','fixed_both',*remaining]
    run_spread_gpu.main()
    path=out/'results.json'
    if not path.exists():return
    result=json.loads(path.read_text())
    model=RWKVMuxModel(result['arguments']['seed'],opts.mode,result['arguments']['ffn_charts'])
    result['arguments'].update(rwkv_mux_mode=opts.mode,side_width=2048,history_mode='previous_layers_post_attention_and_post_ffn')
    result['architecture']='Demuxed-history baseline (4 KS2 FFNs, 64 rank-8 charts, 96D main, 2048D side); RWKV mux replacement: '+opts.mode
    result['parameter_breakdown']=parameter_breakdown(model)
    result['rwkv_replacements']=model.replacements
    result['coding_protocol']=dict(
        decoder='Unchanged HistoryModel: input widths 2144/2336/2528/2720, fixed 512-coordinate mux, KS2 geometry, demux to 2144 with zero-init learned gains. Separate prior post-attention and post-FFN main states in history.',
        rwkv='Selected linear types replaced across all layers where present. Signed/permuted rank-8 Hadamard views folded across all inputs. Unit row norms; every input present in every output coordinate. Learned per-output gains, no learned dense matrix.',
        initialization='Each gain starts at the original dense row L2 norm; zero rows remain zero. Unreplaced parameters and all decoder codes identical to baseline. Whole-model initial logits match because attention and decoder residual outputs start at zero.',
        optimizer='RWKV mux gains retain original dense-weight base LR and WD (0.001 cosine to 0.0002, WD 0.01); decoder demux gains retain 5.5x LR and inverse WD.',
        implementation='Fixed matrices are precomputed Hadamard codes, applied by F.linear; no optimized butterfly kernel or dynamic modulation.',
        scope='All 46 nn.Linear modules inside RWKV for mode all. Embedding, logits head, decoder geometry/mux/demux, nonlinearities, normalization, time mixing and recurrence remain baseline.')
    for name in ('side_stream.py','side_history.py','rwkv_mux.py','run_rwkv_mux_gpu.py'):
        result['source_hashes'][name]=hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
