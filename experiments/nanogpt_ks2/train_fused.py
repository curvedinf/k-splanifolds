"""Repeat only the KS2 arm using an existing experiment's configuration and streams."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from experiments.nanogpt_ks2 import run


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference-run',type=Path,required=True)
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seeds',type=int,nargs='+',default=[11])
    cli=ap.parse_args()
    reference=json.loads((cli.reference_run/'config.json').read_text())
    cfg=dict(reference['arguments'])
    cfg.update(out=cli.out,data_dir=cli.data_dir,seeds=cli.seeds,ks2_backend='fused',device='cpu')
    args=SimpleNamespace(**cfg)
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
    train,val,manifest=run.data(args.data_dir)
    for name,info in manifest.items():
        if info['sha256']!=reference['data_manifest'][name]['sha256']:raise ValueError('Data hash mismatch')
    prefix=run.prefix_table()
    validation=[torch.from_numpy(np.array(val[i:i+args.sequence+1],dtype=np.int64)) for i in range(0,args.val_tokens,args.sequence)]
    steps=reference['fixed_steps'];results=[]
    for seed in args.seeds:
        stream,offset=run.batches(train,steps,args.sequence,seed)
        result=run.train('ks2',seed,args,steps,stream,prefix,validation,dict(train=offset,validation=0))
        prior=json.loads((cli.reference_run/f'ks2_{seed}.json').read_text())
        result['reference_final_val_loss']=prior['history'][-1]['val_loss']
        result['fused_minus_reference']=result['final_val_loss']-result['reference_final_val_loss']
        results.append(result)
        run.save(args.out/'results.json',results)
    run.save(args.out/'config.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in cfg.items()},fixed_steps=steps,
               reference_run=str(cli.reference_run),data_manifest=manifest,torch=torch.__version__,
               sources={str(p):run.sha(p) for p in [Path(__file__),Path(run.__file__),Path(__file__).parents[1]/'common/ks2_cpu/training.py',Path(__file__).parents[1]/'common/ks2_cpu/backward.cpp']}))
    print('RESULTS',json.dumps(results),flush=True)


if __name__=='__main__':main()
