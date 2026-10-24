"""Capacity-preserving KS3 wall-clock research; the original MLP is fixed."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.common.ks3_int8 import capture_training
from experiments.nanogpt_ks2 import run
from experiments.nanogpt_ks2.model import GPT
from experiments.nanogpt_ks2.optim import Optimizer

MIN_FFN, MAX_FFN = 741400, 743424
WALL_BUDGET = 39.8978326653208


def make_model(cfg, seed, args):
    kind=cfg.get('kind','ks3')
    torch.manual_seed(seed)
    rank=cfg.get('rank',1)
    charts=cfg.get('charts',85 if rank==4 else 337)
    model=GPT(kind,96,ks3_rank=rank,ks3_subspaces=charts,
              ks3_backend=cfg.get('backend','fused'),ks3_nonlinear='fp16')
    if 'banks' in cfg:
        if kind!='ks3' or not cfg.get('whole_graph',False):
            raise ValueError('Mixed-rank research requires KS3 whole-model capture')
        if cfg.get('basis','hermite')!='hermite':
            raise ValueError('Mixed-rank research currently uses the Hermite basis')
        from experiments.nanogpt_ks2.wallclock_cells import MixedKS3
        model.ff=torch.nn.ModuleDict({key:MixedKS3(96,cfg['banks']) for key in model.ff})
        model.config['ks3_banks']=cfg['banks']
    if kind=='ks3':
        with torch.no_grad():
            for cell in model.ff.values():
                cell.coefficients.mul_(cfg.get('coefficient_init',.05)/.05)
                cell.down.weight.mul_(cfg.get('down_scale',1.))
                if 'raw_scale' in cfg:cell.raw_scale.fill_(cfg['raw_scale'])
        if cfg.get('basis','hermite')!='hermite':
            from experiments.nanogpt_ks2.wallclock_cells import change_basis
            for cell in model.ff.values():change_basis(cell,cfg['basis'])
    model=model.to(args.device)
    if cfg.get('window_attention',False):
        if kind!='ks3' or cfg.get('amp',False):
            raise ValueError('Window kernel research uses KS3 with FP32 shared arithmetic')
        from experiments.nanogpt_ks2.wallclock_attention import install_window_attention
        install_window_attention(model)
    if kind=='ks3' and cfg.get('whole_graph',False):
        from experiments.nanogpt_ks2.wallclock_graph import capture_model
        capture_model(model,args.sequence,amp=cfg.get('amp',False))
    elif kind=='ks3' and cfg.get('ffn_graph',True):
        for cell in model.ff.values():
            sample=torch.zeros(1,args.sequence,args.width,device=args.device,requires_grad=True)
            capture_training(cell,sample)
    opt=Optimizer(model,args.lr_scale)
    if kind=='ks3':
        geometry={id(p) for cell in model.ff.values() for p in
                  (cell.coefficients,cell.raw_scale,cell.raw_shift)}
        down={id(cell.down.weight) for cell in model.ff.values()}
        up={id(cell.up.weight) for cell in model.ff.values()}
        for group in opt.adam:
            if id(group['p']) in geometry:group['mul']*=cfg.get('geometry_lr',1.)
        for group in opt.matrices:
            if id(group['p']) in down:group['multiplier']*=cfg.get('down_lr',1.)
            if id(group['p']) in up:group['multiplier']*=cfg.get('up_lr',1.)
        count=sum(p.numel() for p in model.ff.parameters())
        if not MIN_FFN<=count<=MAX_FFN:
            raise ValueError(f'FFN budget violated: {count}, expected [{MIN_FFN}, {MAX_FFN}]')
        if not all(p.requires_grad for p in model.parameters()):
            raise ValueError('All parameters must remain trainable')
        if cfg.get('optimizer_graph',False):
            from experiments.nanogpt_ks2.wallclock_optim import GraphOptimizer
            opt=GraphOptimizer(opt,batch_matrices=cfg.get('batch_optimizer',False),
                               fuse_adam=cfg.get('fused_optimizer_adam',False),
                               gradient_scale=cfg.get('gradient_scale',1.),
                               scale_in_update=cfg.get('scale_in_update',False))
    return model,opt


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--data-dir',type=Path,default=Path('/home/curved/modded-nanogpt-ssinf/data/fineweb10B'))
    cli=ap.parse_args();cfg=json.loads(cli.config.read_text())
    if cfg.get('head_fp16',False) or cfg.get('full_vocab',False) or 'lr_end' in cfg or cfg.get('fused_head_loss',False):
        if cfg.get('kind','ks3')!='ks3':
            raise ValueError('Head precision research must not change the MLP control')
        from functools import partial
        from experiments.nanogpt_ks2.wallclock_loss import loss
        run.loss=partial(loss,head_fp16=cfg.get('head_fp16',False),
                         full_vocab=cfg.get('full_vocab',False),lr_end=cfg.get('lr_end',.3),
                         fused_head_loss=cfg.get('fused_head_loss',False))
        if cfg.get('head_fp16',False):cfg.setdefault('gradient_scale',64.)
    if cfg.get('amp',False) and not (cfg.get('whole_graph',False) and cfg.get('optimizer_graph',False)):
        raise ValueError('AMP research requires whole-model and optimizer graphs')
    gradient_scale=cfg.get('gradient_scale',1.)
    if gradient_scale!=1.:
        if cfg.get('kind','ks3')!='ks3' or not cfg.get('optimizer_graph',False):
            raise ValueError('Loss scaling requires KS3 and the matching optimizer unscale')
        original_loss=run.loss
        def scaled_loss(*args,**kwargs):
            value,lr=original_loss(*args,**kwargs)
            return value*gradient_scale,lr
        run.loss=scaled_loss
    reference=json.loads(Path(run.__file__).with_name('config.json').read_text())
    settings=dict(reference['arguments'])
    steps=cfg.get('steps',500)
    settings.update(device=f'cuda:{cli.gpu}',out=cli.out,data_dir=cli.data_dir,
                    steps=steps,max_seconds=max(3600,steps*.2),eval_every=cfg.get('eval_every',100),
                    lr_scale=cfg.get('lr_scale',.25),sequence=cfg.get('sequence',128))
    if cfg.get('kind','ks3')=='mlp' and (settings['lr_scale']!=.25 or settings['sequence']!=128):
        raise ValueError('The primary MLP control must retain its original learning rate and sequence')
    args=SimpleNamespace(**settings);args.out.mkdir(parents=True,exist_ok=False)
    paths=[Path(run.__file__),Path(__file__),Path(__file__).with_name('model.py'),
           Path(__file__).with_name('optim.py'),Path(__file__).parents[1]/'common/ks3_int8.py',
           Path(__file__).parents[1]/'common/ks3.py']
    paths += [p for p in Path(__file__).parent.glob('wallclock_*.py')]
    source_hashes={}
    repository=Path(__file__).resolve().parents[2]
    for path in paths:
        content=path.read_bytes()
        destination=args.out/'sources'/path.resolve().relative_to(repository)
        destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(content)
        source_hashes[str(path)]=hashlib.sha256(content).hexdigest()
    torch.cuda.set_device(cli.gpu)
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    train,val,manifest=run.data(cli.data_dir)
    assert all(info['sha256']==reference['data_manifest'][name]['sha256'] for name,info in manifest.items())
    prefix=run.prefix_table().to(args.device)
    # Keep the control's exact evaluation context even when training context changes.
    validation=[torch.from_numpy(np.array(val[i:i+129],dtype=np.int64)).to(args.device)
                for i in range(0,args.val_tokens,128)]
    stream,offset=run.batches(train,steps,args.sequence,cli.seed)
    stream=[t.to(args.device) for t in stream]
    probe,_=run.batches(train,1,args.sequence,777)
    last={}
    def make(kind,seed,a):
        model,opt=make_model(cfg,seed,a)
        last['optimizer']=opt
        last['model']=model
        return model,opt
    run.make=make
    kind=cfg.get('kind','ks3')
    pilot=run.pilot(kind,args,probe[0].to(args.device),prefix,validation)
    # The inspection dictionary must not retain a second model/graph during
    # timed setup. Graph wrappers form cycles, so collect them explicitly.
    last.clear()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(args.device)
    budget=cfg.get('wall_budget',WALL_BUDGET)
    meta=dict(candidate=cfg,seed=cli.seed,gpu=cli.gpu,wall_budget_seconds=budget,
              data_manifest=manifest,pilot=pilot,torch=torch.__version__,hip=torch.version.hip,
              sources=source_hashes,sources_snapshot='sources/',
              timing='Timed model setup/capture, all steps, all scheduled validations and checkpoint save; disposable pilot and data loading excluded.',
              eval_every=args.eval_every)
    run.save(args.out/'config.json',meta)
    result=run.train(kind,cli.seed,args,steps,stream,prefix,validation,dict(train=offset,validation=0))
    result.update(candidate=cfg,gpu=cli.gpu,step_ms=1000*result['train_seconds']/steps,
                  within_wall_budget=(result['seconds']<=budget if budget is not None else None),
                  optimizer_graph_fallbacks=getattr(last['optimizer'],'fallbacks',0))
    result['whole_graph_stats']=getattr(last['model'],'_whole_graph_stats',None)
    result['training_tokens']=steps*args.sequence
    result['validation_sequence']=128
    result['peak_allocated_bytes']=torch.cuda.max_memory_allocated(args.device)
    if cfg.get('whole_graph',False):
        assert result['whole_graph_stats']['replays']==steps
    run.save(args.out/'result.json',result)
    print('RESULT',json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)


if __name__=='__main__':main()
