"""Verify official ROSA forward semantics and the declared counterfactual VJP."""
import argparse
import ast
import json
from pathlib import Path
import random
import time
import torch
from rosa_gpu import reference_symbols,reference_forward,reference_credits,rosa


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cpu-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    source=Path(__file__).parent/'source_snapshots/rwkv8_reference/251105_reverse_run.py'
    tree=ast.parse(source.read_text())
    selected=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='samx_qkv_slow']
    scope={};exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),scope)
    official=scope['samx_qkv_slow']
    rng=random.Random(11)
    for length in (1,2,8,17,32,64):
        patterns=[([0]*length,[0]*length,[1]*length),([1]*length,[0]*length,[1]*length),
                  ([i%2 for i in range(length)],[i%2 for i in range(length)],[i%3==0 for i in range(length)])]
        patterns += [tuple([rng.randrange(2) for _ in range(length)] for _ in range(3)) for _ in range(30)]
        for q,k,v in patterns:
            assert reference_symbols(q,k,v)==[2*x-1 for x in official(q,k,v)],(q,k,v)
    if args.cpu_only:
        print(json.dumps(dict(status='PASS',check='exhaustive-definition forward matches pinned official suffix-automaton reference on causal, periodic, no-match and random streams through T=64')))
        return
    errors=[]
    for length,channels in ((1,2),(8,3),(17,2),(64,2)):
        torch.manual_seed(11+length)
        q,k,v=[torch.randn(2,length,channels) for _ in range(3)]
        # Exercise exact-zero signs and long matching suffixes.
        q[0,:,0]=1.;k[0,:,0]=1.;q[1,0,0]=0.;k[1,0,0]=0.
        dy=torch.randn_like(q)
        expected=reference_forward(q,k,v)
        expected_grad=reference_credits(q,k,v,dy)
        device_inputs=[x.cuda().requires_grad_() for x in (q,k,v)]
        actual=rosa(*device_inputs)
        torch.testing.assert_close(actual.cpu(),expected,atol=0,rtol=0)
        actual.backward(dy.cuda())
        for x,g in zip(device_inputs,expected_grad):
            torch.testing.assert_close(x.grad.cpu(),g,atol=3e-5,rtol=3e-5)
        errors.append(max((x.grad.cpu()-g).abs().max().item() for x,g in zip(device_inputs,expected_grad)))
    # Check every trailing-bit position, including bit 63 and zero XOR.
    for position in range(64):
        q=torch.ones(1,64,1,device='cuda');k=q.clone();v=q.clone()
        q[:,position]=-1;v[:,::2]=-1
        torch.testing.assert_close(rosa(q,k,v).cpu(),reference_forward(q,k,v).cpu(),atol=0,rtol=0)
    q,k,v=[torch.randn(4,64,32,device='cuda',requires_grad=True) for _ in range(3)]
    dy=torch.randn_like(q)
    for _ in range(3):
        rosa(q,k,v).backward(dy)
        for x in (q,k,v):x.grad=None
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rosa(q,k,v).backward(dy)
    torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(30):graph.replay()
    torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000/30
    print(json.dumps(dict(status='PASS',gpu_vs_bruteforce_gradient_max_errors=errors,
        kernel_forward_backward_ms=ms,checks=['pinned official forward semantics','full-window untruncated matches',
        'all Q/K/V finite-flip credits with Bernoulli mapping','all 64 bit positions',
        'FP32 GPU graph capture']),indent=2))


if __name__=='__main__':main()
