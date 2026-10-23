"""Saved-checkpoint forward timing: paired inputs, full logits, warmed/interleaved trials.

python experiments/nanogpt_ks2/bench_forward.py --checkpoints runs/nanogpt-ks2
Native export retains FP32 coefficients (the standard QAT export requantizes them).
"""
import argparse
import copy
import json
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.nanogpt_ks2.model import GPT
from experiments.nanogpt_ks2.run import schedule, documents, save
from experiments.common.ks2_cpu.inference import KS2Horner, _state


def load_checkpoint(path, device=None):
    path = Path(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    args = checkpoint['args']
    kind = checkpoint.get('kind', 'mlp' if path.stem.startswith('mlp') else
                          'ks3' if path.stem.startswith('ks3') else 'ks2')
    cfg = checkpoint.get('model_config', dict(kind=kind, width=args['width'], rank=args['rank'],
                         ks3_subspaces=args.get('ks3_subspaces'), ks3_rank=args.get('ks3_rank',1),
                         ks3_backend=args.get('ks3_backend','fused'), ks3_nonlinear=args.get('ks3_nonlinear','fp16')))
    fused_ks3 = kind == 'ks3' and cfg.get('ks3_backend','fused') == 'fused'
    if device is None:
        device = 'cuda' if fused_ks3 else 'cpu'
    if fused_ks3 and (torch.device(device).type != 'cuda' or not torch.cuda.is_available()):
        raise ValueError('This fused KS3 checkpoint requires a CUDA/ROCm GPU')
    model = GPT(**cfg)
    model.load_state_dict(checkpoint['model'])
    # Frequencies are checkpoint buffers; the original YaRN scale is a Python float.
    # Reconstruct only that scalar from the saved run length, without rotating twice.
    old = [16, 16, 48]
    steps = checkpoint['history'][-1]['step']
    for step in range(steps):
        window = schedule(step, steps)[0]
        new = [window[0], window[0], window[1]]
        for yarn, a, b in zip((model.yarn, model.yarn_paired_head, model.yarn_wide), old, new):
            if a != b:
                import math
                yarn.attn_scale *= 1 + .2*math.log(b/a)
        old = new
    return model.to(device).eval(), checkpoint


class NativeFP32(nn.Module):
    def __init__(self, source, threads):
        super().__init__()
        self.cell = KS2Horner(source, threads)
        s = _state(source)
        a,b,c,d,e,f = s['geo_t'].unbind(1)
        self.cell.a.copy_(torch.stack((a,c,-3*a+3*b-2*c-d,2*a-2*b+c+d,e,f),1))
        a,b,c,d = s['geo_x'].unbind(2)
        self.cell.q.copy_(torch.stack((a-c,b-3*d,2*c,4*d),1))

    def forward(self, x):
        return self.cell(x.reshape(-1,x.shape[-1])).reshape(x.shape)


@torch.inference_mode()
def validate(model, batches):
    model.eval()
    total, count = 0., 0
    for tokens in batches:
        x, y = tokens[:-1], tokens[1:]
        hidden = model(x,80,176,documents(x))
        for lo in range(0,len(x),64):
            logits = 23*torch.sigmoid((F.linear(hidden[lo:lo+64],model.head_weight())+5)/7.5)
            total += F.cross_entropy(logits,y[lo:lo+64],reduction='sum').item()
        count += len(x)
    return total/count


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoints',type=Path,required=True)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--data-dir',type=Path)
    ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--lengths',type=int,nargs='+',default=[128,512])
    ap.add_argument('--repeats',type=int,default=15)
    ap.add_argument('--warmup',type=int,default=3)
    args=ap.parse_args()
    out=args.out or args.checkpoints/'forward.json'
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    config=json.loads((args.checkpoints/'config.json').read_text())
    data_path=(args.data_dir or Path(config['arguments']['data_dir']))/'fineweb_val_000000.bin'
    data=np.memmap(data_path,dtype='<u2',mode='r',offset=1024)
    seq=config['arguments']['sequence']
    val=[torch.from_numpy(np.array(data[i:i+seq+1],dtype=np.int64)) for i in range(0,config['arguments']['val_tokens'],seq)]
    timings=[];checks=[];setups=[]
    with torch.inference_mode():
        for seed in config['arguments']['seeds']:
            mlp, mc=load_checkpoint(args.checkpoints/f'mlp_{seed}.pt')
            ks, kc=load_checkpoint(args.checkpoints/f'ks2_{seed}.pt')
            for name, model, ck in [('mlp',mlp,mc),('ks2',ks,kc)]:
                ce=validate(model,val);model.eval()
                error=abs(ce-ck['history'][-1]['val_loss'])
                if error>1e-6:raise RuntimeError(f'Checkpoint reconstruction mismatch {name}: {error}')
                checks.append(dict(seed=seed,variant=name,loaded_ce=ce,recorded_ce_error=error))
            compiled=copy.deepcopy(mlp)
            for key,ff in list(compiled.ff.items()):compiled.ff[key]=torch.compile(ff,fullgraph=True,dynamic=False)
            native=copy.deepcopy(ks)
            for key,ff in list(native.ff.items()):native.ff[key]=NativeFP32(ff,args.threads)
            native_ce=validate(native,val)
            checks.append(dict(seed=seed,variant='ks2_native',val_ce=native_ce,ce_delta=native_ce-kc['history'][-1]['val_loss']))
            models={'mlp_eager':mlp,'mlp_compiled_ffn':compiled,'ks2_eager':ks,'ks2_native':native}
            for length in args.lengths:
                x=torch.from_numpy(np.array(data[:length],dtype=np.int64));doc=documents(x)
                references={key: m(x,80,176,doc) for key,m in [('mlp',mlp),('ks2',ks)]}
                for name,m in models.items():
                    start=time.perf_counter()
                    for _ in range(args.warmup):
                        hidden=m(x,80,176,doc)
                        F.linear(hidden,m.head_weight())
                    setups.append(dict(seed=seed,length=length,variant=name,seconds=time.perf_counter()-start))
                    expected=references['mlp' if name.startswith('mlp') else 'ks2']
                    err=(hidden-expected).abs().max().item()
                    logits=F.linear(hidden,m.head_weight());ref_logits=F.linear(expected,m.head_weight())
                    checks.append(dict(seed=seed,length=length,variant=name,max_hidden_error=err,
                                       max_logit_error=(logits-ref_logits).abs().max().item()))
                    if err>1e-3:raise RuntimeError(f'Forward mismatch {name}: {err}')
                trials=[(name,scope) for name in models for scope in ['body','full_logits']]
                rng=random.Random(seed+length)
                for repeat in range(args.repeats):
                    rng.shuffle(trials)
                    for name,scope in trials:
                        m=models[name]
                        start=time.perf_counter()
                        hidden=m(x,80,176,doc)
                        if scope=='full_logits': result=23*torch.sigmoid((F.linear(hidden,m.head_weight())+5)/7.5)
                        elapsed=time.perf_counter()-start
                        timings.append(dict(seed=seed,length=length,variant=name,scope=scope,repeat=repeat,ms=elapsed*1000))
                print('BENCH',seed,length,flush=True)
            del mlp,ks,compiled,native,models
    summary=[]
    for length in args.lengths:
        for scope in ['body','full_logits']:
            for variant in ['mlp_eager','mlp_compiled_ffn','ks2_eager','ks2_native']:
                seed_medians=[statistics.median(r['ms'] for r in timings if r['length']==length and r['scope']==scope and r['variant']==variant and r['seed']==seed) for seed in config['arguments']['seeds']]
                summary.append(dict(length=length,scope=scope,variant=variant,mean_seed_median_ms=statistics.mean(seed_medians),seed_medians_ms=seed_medians))
    result=dict(protocol=dict(threads=args.threads,repeats=args.repeats,warmup=args.warmup,
                             batch=1,lengths=args.lengths,precision='FP32',compile='FFN only, both GEMMs and ReLU², MLP control',
                             full_logits='body + all-token full-vocabulary projection + softcap; no loss or backward',
                             native='unquantized FP32 coefficients; rational tanh and refined reciprocal',
                             order='randomly interleaved backend/scope per repeat',setup_excluded=True),
                summary=summary,checks=checks,setups=setups,timings=timings)
    save(out,result);print('SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__':main()
