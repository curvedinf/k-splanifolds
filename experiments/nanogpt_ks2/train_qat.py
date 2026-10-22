"""Matched 500-step KS2 QAT with padded GS128 ternary weights and FP32 masters."""
import argparse
import json
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.nn import functional as F
from experiments.nanogpt_ks2 import run
from experiments.nanogpt_ks2.tq1_gs128 import roundtrip
from experiments.nanogpt_ks2.quantize_tq1 import quantize
from experiments.nanogpt_ks2.quantize_controls import NAMES as CONTROLS
from experiments.common.ks2_cpu.training import _Cell, NAMES, fuse_training


class Ternary(torch.autograd.Function):
    @staticmethod
    def forward(ctx,w):
        flat=w.reshape(w.shape[0],-1)
        width=flat.shape[1]
        blocks=F.pad(flat,(0,(-width)%128)).reshape(-1,128)
        scale=blocks.abs().amax(1,keepdim=True)
        inv=torch.where(scale!=0,scale.reciprocal(),0.)
        z=blocks*inv
        q=z.sign()*(z.abs()+.5).floor()*scale.half().float()
        return q.reshape(flat.shape[0],-1)[:,:width].reshape_as(w)

    @staticmethod
    def backward(ctx,grad):
        return grad


def weights(ff):
    q=Ternary.apply;s=ff.geo_spine.shape[0]
    t=q(torch.cat((ff.geo_spine,ff.geo_t_basis_edge.reshape(s,6,-1)),-1))
    x=q(ff.geo_x_basis_edge.permute(0,2,1,3))
    result={'down.weight':q(ff.down.weight),'up.weight':q(ff.up.weight),
            'geo_spine':t[:,:,:8],'geo_t_basis_edge':t[:,:,8:].reshape_as(ff.geo_t_basis_edge),
            'geo_x_basis_edge':x.permute(0,2,1,3)}
    for name,attr in CONTROLS.items():
        w=getattr(ff,attr)
        result[attr]=q(w.reshape(s,-1) if name in ('distance','importance') else w.reshape(1,-1)).reshape_as(w)
    assert set(result)==set(dict(ff.named_parameters()))
    return result


def forward(self,inputs):
    w=weights(self)
    return _Cell.apply(inputs,w['down.weight'],w['up.weight'],self._backward_threads,*(w[n] for n in NAMES))


def install(model,threads):
    for ff in model.ff.values():
        fuse_training(ff,threads)
        ff.forward=MethodType(forward,ff)


@torch.no_grad()
def export(model):
    stats=quantize(model,'tq1_geometry_projections',roundtrip=roundtrip)
    for key,ff in model.ff.items():
        for name,attr in CONTROLS.items():
            w=getattr(ff,attr)
            view=w.reshape(w.shape[0],-1) if name in ('distance','importance') else w.reshape(1,-1)
            q,stats[key][attr]=roundtrip(view);w.copy_(q.reshape_as(w))
    return sum(x['packed_bytes'] for ff in stats.values() for x in ff.values())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference-run',type=Path,required=True)
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seeds',type=int,nargs='+',default=[11,22,33])
    cli=ap.parse_args();ref=json.loads((cli.reference_run/'config.json').read_text())
    cfg=dict(ref['arguments']);cfg.update(out=cli.out,data_dir=cli.data_dir,seeds=cli.seeds,ks2_backend='fused',qat='tq1_gs128')
    args=SimpleNamespace(**cfg);args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
    train,val,manifest=run.data(args.data_dir)
    assert all(info['sha256']==ref['data_manifest'][name]['sha256'] for name,info in manifest.items())
    prefix=run.prefix_table()
    validation=[torch.from_numpy(np.array(val[i:i+args.sequence+1],dtype=np.int64)) for i in range(0,args.val_tokens,args.sequence)]
    make=run.make
    def make_qat(kind,seed,args):
        model,opt=make(kind,seed,args);install(model,args.threads)
        return model,opt
    run.make=make_qat
    steps=ref['fixed_steps'];results=[]
    run.save(args.out/'config.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in cfg.items()},fixed_steps=steps,data_manifest=manifest,
        protocol='From scratch, matched seeds/streams/schedule; all FFN weights GS128 ternary from step zero; identity STE; detached absmax FP16 scales; FP32 masters, optimizer, activations and backward; fused KS2 analytic backward',
        torch=torch.__version__,sources={p.name:run.sha(p) for p in (Path(__file__),Path(run.__file__),Path(__file__).with_name('tq1_gs128.py'))}))
    for seed in args.seeds:
        stream,offset=run.batches(train,steps,args.sequence,seed)
        result=run.train('ks2',seed,args,steps,stream,prefix,validation,dict(train=offset,validation=0))
        from experiments.nanogpt_ks2.bench_forward import load_checkpoint,validate
        model,ck=load_checkpoint(args.out/f'ks2_{seed}.pt')
        expected={key:{name:w.clone() for name,w in weights(ff).items()} for key,ff in model.ff.items()}
        result['packed_ffn_bytes']=export(model)
        for key,ff in model.ff.items():
            for name,w in ff.named_parameters():torch.testing.assert_close(w,expected[key][name],rtol=0,atol=0)
        result['packed_export_ce']=validate(model,validation)
        result['export_minus_qat_ce']=result['packed_export_ce']-result['final_val_loss']
        assert abs(result['export_minus_qat_ce'])<1e-5
        prior=json.loads((cli.reference_run/f'ks2_{seed}.json').read_text())
        result['reference_fp32_ce']=prior['history'][-1]['val_loss']
        result['delta_from_fp32']=result['packed_export_ce']-result['reference_fp32_ce']
        results.append(result);run.save(args.out/'results.json',results)
        print('EXPORT',json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)


if __name__=='__main__':main()
