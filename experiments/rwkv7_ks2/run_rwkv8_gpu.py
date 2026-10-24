"""Matched demuxed-history training with the official-style RWKV7+ROSA hybrid."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import run
import run_spread_gpu
from rwkv8_rosa import MODES,RWKV8Model,make_optimizer,parameter_breakdown


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mode',choices=MODES,required=True)
    opts,remaining=parser.parse_known_args()
    out=Path(remaining[remaining.index('--out')+1])
    def factory(seed,variant,**kwargs):
        assert kwargs.get('width',96)==96 and kwargs.get('layers',4)==4
        return RWKV8Model(seed,opts.mode,kwargs.get('ffn_charts',64))
    run.Model,run.make_optimizer=factory,make_optimizer
    run_spread_gpu.install=lambda model,coding:model
    sys.argv=[sys.argv[0],'--coding','fixed_both',*remaining]
    run_spread_gpu.main()
    path=out/'results.json'
    if not path.exists():return
    result=json.loads(path.read_text())
    model=RWKV8Model(result['arguments']['seed'],opts.mode,result['arguments']['ffn_charts'])
    result['arguments'].update(rwkv8_mode=opts.mode,side_width=2048,
        history_mode='previous_layers_post_attention_and_post_ffn')
    result['architecture']='Demuxed-history KS2 decoders with attention variant '+opts.mode
    result['parameter_breakdown']=parameter_breakdown(model)
    result['rwkv8_protocol']=dict(
        source='BlinkDL/RWKV-LM commit 20be0f8cf0bdf506d8c0030c87a9a3416f95b143, RWKV-v8/251105_reverse_run.py',
        architecture='RWKV-7 recurrence plus parallel 1-bit QKV ROSA, each with its own LayerNorm on the same 96D main input. Add both updates before the unchanged KS2 decoder. History stores the combined post-attention state.',
        rosa='32 binary Q/K/V streams, 96-to-32 biased projections, three 96D time-mix vectors, 32 learned payload scales, biased 32-to-96 output. Longest matching Q suffix in strictly previous K; latest endpoint wins; read successor V. No match and matched zero both map to negative payload, matching the pinned hybrid.',
        backward='Exact single-bit counterfactual Q/K/V credits across the full 64-token window, multiplied by sigmoid(logit)*(1-sigmoid(logit)), tau=1. This is a specified discrete surrogate, not the analytic derivative of hard thresholds. GPU implementation verified against brute force.',
        reduced_ranks=dict(decay=16,rate=16,value=16,gate=40),original_ranks=dict(decay=32,rate=32,value=32,gate=64),
        initialization='Original decoder/embedding/head weights paired. Reduced controllers keep the leading columns/rows of original factors. ROSA QKV use PyTorch Linear initialization, payload scales one, time mix zero, output weight/bias zero to preserve initial logits.',
        unchanged='All four 64-rank8 KS2 decoders, 2048D side residual and both historical main states per previous layer. No Hadamard replacements in RWKV or ROSA. Same 15k-step FP32 data/optimizer schedule.',
        implementation='Short-window GPU bit-packed exact suffix comparisons and full single-bit flip enumeration; not an asymptotically optimized long-context ROSA automaton.',
        qualification='A local width/budget adaptation of the official hybrid demonstration with an explicitly supplied trainable backward, not an official pretrained RWKV-8 model.')
    for name in ('side_stream.py','side_history.py','rwkv_mux.py','rwkv8_rosa.py','rosa_gpu.py','run_rwkv8_gpu.py'):
        result['source_hashes'][name]=hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
