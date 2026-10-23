"""Isolated FFN forward/backward latency; full training results use compare_ks3.

The optional --baseline points to an earlier ks3_int8.py source snapshot.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import time

import torch

from experiments.common.ks3_int8 import capture_training, fuse_training
from experiments.nanogpt_ks2.model import MLP, make_ks3
from experiments.nanogpt_ks2.run import sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--baseline',type=Path)
    ap.add_argument('--device',default='cuda:0')
    cli=ap.parse_args()
    torch.cuda.set_device(cli.device);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(123)
    cells={'mlp':MLP(96,352).to(cli.device)}
    source=make_ks3(96,352,backend='torch').to(cli.device)
    torch.nn.init.normal_(source.up.weight,std=.02)
    for name in ('ks3_optimized','ks3_compiled','ks3_graph'):
        cell=make_ks3(96,352,backend='torch').to(cli.device)
        cell.load_state_dict(source.state_dict());fuse_training(cell)
        if name.endswith('compiled'):cell.compile(fullgraph=True,dynamic=False)
        cells[name]=cell
    if cli.baseline:
        spec=importlib.util.spec_from_file_location('ks3_baseline',cli.baseline)
        baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
        cell=make_ks3(96,352,backend='torch').to(cli.device)
        cell.load_state_dict(source.state_dict());baseline.fuse_training(cell)
        cells['ks3_initial']=cell
    torch.nn.init.normal_(cells['mlp'].up.weight,std=.02)
    x=torch.randn(128,96,device=cli.device,requires_grad=True)
    grad=torch.randn_like(x)
    capture_training(cells['ks3_graph'],x)
    def step(cell):
        return torch.autograd.grad(cell(x),(x,*cell.parameters()),grad)
    for cell in cells.values():
        for _ in range(20):step(cell)
    samples={name:[] for name in cells}
    for repeat in range(5):
        names=list(cells)
        if repeat%2:names.reverse()
        for name in names:
            torch.cuda.synchronize();start=time.perf_counter()
            for _ in range(100):step(cells[name])
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter()-start)*10)
    result={'shape':[128,96],'charts':337,'rank':1,'gpu':torch.cuda.get_device_name(),
            'torch':torch.__version__,'precision':'MLP FP32; KS3 GS32 W8A8 with FP16 scales/nonlinearities, FP32 VJP',
            'measurement':'Forward + backward, all input/weight gradients; no optimizer; median of five 100-call blocks after warmup.',
            'wall_ms':{name:dict(median=statistics.median(values),samples=values) for name,values in samples.items()},
            'kernel_sha256':sha(Path(__file__).parents[1]/'common/ks3_int8.py')}
    if cli.baseline:result['baseline_sha256']=sha(cli.baseline)
    cli.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
