"""Matched FP32 FFN and whole-model forward/backward timing on saved checkpoints."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.nn import functional as F
from experiments.nanogpt_ks2.bench_forward import load_checkpoint, documents
from experiments.common.ks2_cpu.training import fuse_training


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoints',type=Path,required=True)
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--lengths',nargs='+',type=int,default=[128,512])
    ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--repeats',type=int,default=25)
    ap.add_argument('--scope',choices=['ffn','full','both'],default='both')
    ap.add_argument('--no-compile',action='store_true')
    args=ap.parse_args()
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest');torch.manual_seed(111)
    mlp,_=load_checkpoint(args.checkpoints/'mlp_11.pt')
    ks,_=load_checkpoint(args.checkpoints/'ks2_11.pt')
    fused=copy.deepcopy(ks)
    for ff in fused.ff.values():fuse_training(ff,args.threads)
    models={'mlp_eager':mlp,'ks2_eager':ks,'ks2_fused':fused}
    if not args.no_compile:
        for name,source in [('mlp_compiled_ffn',mlp),('ks2_compiled_ffn',ks)]:
            m=copy.deepcopy(source)
            for k in list(m.ff):m.ff[k]=torch.compile(m.ff[k],fullgraph=True,dynamic=False)
            models[name]=m
    data=np.memmap(args.data_dir/'fineweb_val_000000.bin',dtype='<u2',mode='r',offset=1024)
    records=[];checks=[];setups=[]
    scopes=['ffn','full'] if args.scope=='both' else [args.scope]
    for length in args.lengths:
        for scope in scopes:
            tokens=torch.from_numpy(np.array(data[:length+1],dtype=np.int64))
            doc=documents(tokens[:-1]); x=torch.randn(length,96,requires_grad=True)
            upstream=torch.randn_like(x)
            modules={name:(next(iter(m.ff.values())) if scope=='ffn' else m) for name,m in models.items()}
            def forward(name):
                m=modules[name]
                if scope=='ffn':return m(x)
                h=m(tokens[:-1],80,176,doc)
                logits=23*torch.sigmoid((F.linear(h,m.head_weight())+5)/7.5)
                return F.cross_entropy(logits,tokens[1:])
            def trial(name):
                modules[name].zero_grad(set_to_none=True);x.grad=None
                start=time.perf_counter();result=forward(name);middle=time.perf_counter()
                result.backward(upstream if scope=='ffn' else None)
                end=time.perf_counter()
                return {'forward_ms':(middle-start)*1000,'backward_ms':(end-middle)*1000,'total_ms':(end-start)*1000}
            for name in models:
                start=time.perf_counter()
                for _ in range(4):trial(name)
                setups.append(dict(variant=name,length=length,scope=scope,seconds=time.perf_counter()-start))
                print('READY',name,length,scope,flush=True)
            # Identical checkpoint and upstream gradient: validate the fused replacement.
            expected=forward('ks2_eager').detach();actual=forward('ks2_fused').detach()
            torch.testing.assert_close(actual,expected,atol=3e-5,rtol=2e-4)
            checks.append(dict(length=length,scope=scope,max_output_error=(expected-actual).abs().max().item()))
            rng=random.Random(111+length)
            for repeat in range(args.repeats):
                order=list(models);rng.shuffle(order)
                for name in order:records.append(dict(variant=name,length=length,scope=scope,repeat=repeat,**trial(name)))
            print('DONE',length,scope,flush=True)
    summary=[]
    for scope in scopes:
        for length in args.lengths:
            for name in models:
                rows=[r for r in records if r['scope']==scope and r['length']==length and r['variant']==name]
                summary.append(dict(variant=name,scope=scope,length=length,**{key:statistics.median(r[key] for r in rows) for key in ['forward_ms','backward_ms','total_ms']}))
    result=dict(protocol=dict(seed=11,precision='FP32',threads=args.threads,warmups=4,repeats=args.repeats,
                             compile='FFN only, setup excluded',full='full-vocabulary CE, final windows, no optimizer step',
                             ffn='first checkpoint FFN, fixed random input and upstream gradient',
                             order='randomly interleaved',gradient_reset='set_to_none outside timing',torch=torch.__version__),
                checks=checks,setups=setups,summary=summary,trials=records)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
