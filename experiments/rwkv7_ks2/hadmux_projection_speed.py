"""Fixed Hadamard versus trainable matrices at the SAME projection dimensions.

This is an isolated operator probe, not another trained MLP control.
"""
import argparse
import json
from pathlib import Path
import random
import statistics as st
import torch
import triton.testing
from torch.nn import functional as F
from hadmux_codes import make_codes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--features', type=int, nargs='+', default=[512,2048,6560])
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(5829)
    cases = []
    for features in args.features:
        for direction, width in [('input',2720),('output',2144)]:
            codes = make_codes(width,features//8,8)[0].flatten(0,1)
            matrix = codes if direction=='input' else codes.T.contiguous()
            if direction=='output':
                matrix = matrix/matrix.square().sum(1,keepdim=True).sqrt()
            matrix = matrix.cuda()
            x = torch.randn(256,matrix.shape[1],device='cuda')
            upstream = torch.randn(256,matrix.shape[0],device='cuda')
            observed = []
            for kind in ('fixed','learned'):
                inputs = x.detach().clone().requires_grad_(True)
                weight = matrix.detach().clone().requires_grad_(kind=='learned')
                gain = torch.ones(matrix.shape[0],device='cuda',requires_grad=True) if direction=='output' and kind=='fixed' else None
                targets = (inputs,weight) if kind=='learned' else ((inputs,gain) if gain is not None else (inputs,))
                def operation(inputs=inputs,weight=weight,gain=gain,targets=targets,upstream=upstream):
                    output=F.linear(inputs,weight)
                    if gain is not None:output=output*gain
                    return torch.autograd.grad(output,targets,grad_outputs=upstream)
                gradients=operation()
                assert all(torch.isfinite(g).all() for g in gradients)
                observed.append(gradients[0].detach().clone())
                cases.append(dict(features=features,direction=direction,kind=kind,operation=operation))
            torch.testing.assert_close(*observed,atol=3e-5,rtol=3e-5)
    rows=[]
    rng=random.Random(5829)
    for repeat in range(5):
        order=cases.copy();rng.shuffle(order)
        for case in order:
            ms=triton.testing.do_bench_cudagraph(case['operation'],rep=100)
            rows.append(dict(features=case['features'],direction=case['direction'],kind=case['kind'],repeat=repeat,ms=ms))
        print(json.dumps(dict(event='projection_timing_round',repeat=repeat,cases=len(cases))),flush=True)
    summary=[]
    for case in cases:
        values=[r['ms'] for r in rows if (r['features'],r['direction'],r['kind'])==(case['features'],case['direction'],case['kind'])]
        summary.append(dict(features=case['features'],direction=case['direction'],kind=case['kind'],median_ms=st.median(values),min_ms=min(values),max_ms=max(values)))
    result=dict(status='PASS',protocol='One MI100, FP32,256positions;five randomized captured rounds. Input projection2720→F;outputF→2144. ExactsameinitialHadamardmatrix in fixed and learned cases. Forward and input gradients included in both; learned case also computes matrix gradients. Fixed output also computes per-output gain gradients. No attention, optimizer or full-model timing; these are not additional quality/control experiments.',rows=rows,summary=summary)
    args.out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
