"""Fixed-step FP32 convergence comparison of reduced record-92 MLP and quarter-budget KS2.

python experiments/nanogpt_ks2/run.py --data-dir data/fineweb10B --out runs/nanogpt-ks2
A timing pilot sets ONE common step count (at most --steps) before either run.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from urllib.request import urlopen

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.nanogpt_ks2.model import GPT, UPSTREAM
from experiments.nanogpt_ks2.optim import Optimizer

DATA_REVISION = '889765ea1f903759787add96995d81171b632d0c'


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while b := f.read(1024*1024): h.update(b)
    return h.hexdigest()


def data(root):
    root.mkdir(parents=True, exist_ok=True)
    arrays, manifest = [], {}
    for name in ('fineweb_train_000001.bin', 'fineweb_val_000000.bin'):
        p = root/name
        url = f'https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/{DATA_REVISION}/{name}'
        if not p.exists():
            with urlopen(url, timeout=120) as response, p.with_suffix('.download').open('wb') as out:
                while b := response.read(1024*1024): out.write(b)
            p.with_suffix('.download').replace(p)
        header = np.fromfile(p, dtype='<i4', count=256)
        if len(header) != 256 or header[0] != 20240520 or header[1] != 1 or p.stat().st_size != 1024 + int(header[2])*2:
            raise ValueError(f'Invalid FineWeb shard: {p}')
        manifest[name] = dict(sha256=sha(p), tokens=int(header[2]), url=url)
        arrays.append(np.memmap(p, dtype='<u2', mode='r', offset=1024, shape=(int(header[2]),)))
    return *arrays, manifest


def prefix_table():
    import tiktoken
    ranks = tiktoken.get_encoding('gpt2')._mergeable_ranks
    table, stack = torch.full((50257,), -1, dtype=torch.long), []
    for text, token in sorted(ranks.items()):
        while stack and not text.startswith(stack[-1][0]): stack.pop()
        if stack: table[token] = stack[-1][1]
        stack.append((text, token))
    return table


def schedule(step, total):
    p = step / total
    if p < .2854: window, mtp, prefix = (16, 48), [1., .5, .25*(1-p/.2854)], .25
    elif p < .6063:
        f = (p-.2854)/.3209
        window, mtp, prefix = (48, 112), [1., .5*(1-f)], [.2, .15, .1, .05][min(3, int(f*4))]
    else: window, mtp, prefix = (80, 176), [1.], 0.
    lr = 1. if p < .2 else 1. - .7*(p-.2)/.8
    candidates = 10240 if p < .6063 else 14336 if p < .81 else 24576 if p < .94 else 0
    return window, mtp, prefix, lr, candidates


def batches(tokens, count, length, seed):
    # One deterministic contiguous stream per seed, EOT-aligned; never replayed.
    rng = np.random.default_rng(seed + 10000)
    offset = int(rng.integers(0, min(1000000, len(tokens)//4)))
    ends = np.flatnonzero(tokens[offset:offset+10000] == 50256)
    if len(ends): offset += int(ends[0])
    if offset + count*length + 3 > len(tokens): raise ValueError('Insufficient tokens without replay')
    return [torch.from_numpy(np.array(tokens[offset+i*length:offset+(i+1)*length+3], dtype=np.int64)) for i in range(count)], offset


def documents(x):
    return (x == 50256).long().cumsum(0)


def loss(model, tokens, prefix, step, steps, seed, windows):
    if getattr(model, '_ks3_compile', False):
        torch.compiler.cudagraph_mark_step_begin()
    window, mtp, prefix_weight, lr, candidates = schedule(step, steps)
    for yarn, old, new in zip((model.yarn, model.yarn_paired_head, model.yarn_wide), windows, (window[0], window[0], window[1])):
        if old != new: yarn.apply_window(old, new)
    windows[:] = [window[0], window[0], window[1]]
    x = tokens[:-3]
    hidden = model(x, *window, documents(x))
    targets = tokens[1:len(x)+1]
    all_targets = tokens[1:]
    pref = prefix[all_targets]
    generator = torch.Generator(device=tokens.device).manual_seed(seed*100000 + step)
    if candidates:
        ids = torch.unique(torch.cat((all_targets, pref[pref >= 0], torch.randperm(50257, generator=generator, device=tokens.device)[:candidates])))
        weight = model.head_weight()[ids]
    else: ids, weight = None, model.head_weight()
    logits = 23*torch.sigmoid((F.linear(hidden, weight)+5)/7.5)
    logp = F.log_softmax(logits, dim=-1)
    rows = torch.arange(len(x), device=x.device)
    result = logp.new_zeros(())
    for shift, coefficient in enumerate(mtp):
        target = tokens[1+shift:1+shift+len(x)]
        mapped = torch.searchsorted(ids, target) if ids is not None else target
        result = result - coefficient*logp[rows, mapped].mean()
    if prefix_weight:
        target = prefix[targets]
        valid = target >= 0
        mapped = torch.searchsorted(ids, target.clamp_min(0)) if ids is not None else target.clamp_min(0)
        result = result - prefix_weight*(logp[rows, mapped]*valid).mean()
    return result, lr


@torch.no_grad()
def validate(model, batches):
    model.eval()
    total, count = 0., 0
    for tokens in batches:
        if getattr(model, '_ks3_compile', False):
            torch.compiler.cudagraph_mark_step_begin()
        x, y = tokens[:-1], tokens[1:]
        hidden = model(x, 80, 176, documents(x))
        for lo in range(0, len(x), 64):
            logits = 23*torch.sigmoid((F.linear(hidden[lo:lo+64], model.head_weight())+5)/7.5)
            total += F.cross_entropy(logits, y[lo:lo+64], reduction='sum').item()
        count += len(x)
    model.train()
    return total/count


def make(kind, seed, args):
    torch.manual_seed(seed)
    if kind == 'ks3' and getattr(args, 'ks3_backend', 'fused') == 'fused' and torch.device(args.device).type != 'cuda':
        raise ValueError('Fused KS3 needs a GPU; use --device cuda or --ks3-backend torch')
    model = GPT(kind, args.width, rank=args.rank,
                ks3_subspaces=getattr(args, 'ks3_subspaces', None),
                ks3_rank=getattr(args, 'ks3_rank', 1),
                ks3_backend=getattr(args, 'ks3_backend', 'fused'),
                ks3_nonlinear=getattr(args, 'ks3_nonlinear', 'fp16')).to(args.device)
    if kind == 'ks2' and getattr(args, 'ks2_backend', 'torch') == 'fused':
        from experiments.common.ks2_cpu.training import fuse_training
        for ff in model.ff.values(): fuse_training(ff, args.threads)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    mode = getattr(args, 'ks3_compile', 'off')
    if kind == 'ks3' and getattr(args, 'ks3_graph', False):
        if mode != 'off':
            raise ValueError('Choose graph replay or torch.compile, not both')
        from experiments.common.ks3_int8 import capture_training
        for cell in model.ff.values():
            sample = torch.zeros(1,args.sequence,args.width,device=args.device,requires_grad=True)
            capture_training(cell, sample)
    if kind == 'ks3' and mode != 'off':
        for cell in model.ff.values():
            cell.compile(mode=mode, fullgraph=True, dynamic=False)
        model._ks3_compile = True
    return model, Optimizer(model, args.lr_scale)


def sync(args):
    device = torch.device(args.device)
    if device.type == 'cuda': torch.cuda.synchronize(device)


def pilot(kind, args, batch, prefix, validation):
    sync(args)
    start = time.perf_counter()
    model, opt = make(kind, 777, args)
    windows = [16, 16, 48]
    samples = []
    for step in (0, 10, 30, 65, 85, 95):
        sync(args); t = time.perf_counter()
        opt.zero_grad()
        value, lr = loss(model, batch, prefix, step, 100, 777, windows)
        value.backward(); opt.step(step/100, lr)
        sync(args); samples.append(time.perf_counter()-t)
    t = time.perf_counter(); validate(model, validation); sync(args)
    val_seconds = time.perf_counter()-t
    result = dict(step_seconds=max(samples[1:]), validation_seconds=val_seconds,
                  setup_seconds=time.perf_counter()-start-sum(samples)-val_seconds)
    print('PILOT', kind, json.dumps(result), flush=True)
    return result


def train(kind, seed, args, steps, stream, prefix, val, offsets):
    sync(args); start = time.perf_counter()
    model, opt = make(kind, seed, args)
    windows, history, train_seconds = [16, 16, 48], [], 0.
    counts = dict(total=sum(p.numel() for p in model.parameters()), ffn=sum(p.numel() for p in model.ff.parameters()))
    for step in range(steps+1):
        if step == 0 or step % args.eval_every == 0 or step == steps:
            value = validate(model, val)
            if not math.isfinite(value): raise RuntimeError('Nonfinite validation loss')
            history.append(dict(step=step, tokens=step*args.sequence, val_loss=value, wall_seconds=time.perf_counter()-start, train_seconds=train_seconds))
            print(kind, seed, json.dumps(history[-1]), flush=True)
            save(args.out/f'{kind}_{seed}.json', dict(counts=counts, offsets=offsets, history=history))
        if step == steps: break
        if time.perf_counter()-start > args.max_seconds-5:
            raise RuntimeError('Time guard: run incomplete; reduce common --steps and rerun both models')
        sync(args); t = time.perf_counter()
        opt.zero_grad()
        value, lr = loss(model, stream[step], prefix, step, steps, seed, windows)
        if not torch.isfinite(value): raise RuntimeError(f'Nonfinite loss {kind} step {step}')
        value.backward()
        opt.step(step/steps, lr)
        sync(args); train_seconds += time.perf_counter()-t
    torch.save(dict(kind=kind, model_config=model.config, model=model.state_dict(),
                    args=vars(args), history=history), args.out/f'{kind}_{seed}.pt')
    elapsed = time.perf_counter()-start
    if elapsed > args.max_seconds: raise RuntimeError('Run exceeded time cap; lower common step count')
    return dict(kind=kind, seed=seed, steps=steps, counts=counts, final_val_loss=history[-1]['val_loss'],
                initial_val_loss=history[0]['val_loss'], seconds=elapsed, train_seconds=train_seconds,
                history=history)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, default=Path('data/fineweb10B'))
    ap.add_argument('--out', type=Path, default=Path('runs/nanogpt-ks2'))
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--max-seconds', type=float, default=300)
    ap.add_argument('--sequence', type=int, default=128)
    ap.add_argument('--val-tokens', type=int, default=2048)
    ap.add_argument('--eval-every', type=int, default=100)
    ap.add_argument('--width', type=int, default=96)
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--lr-scale', type=float, default=.25)
    ap.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33])
    ap.add_argument('--device', default='cpu', help='cpu, cuda, or cuda:N')
    ap.add_argument('--ks2-backend', choices=['torch','fused'], default='torch')
    ap.add_argument('--models', nargs='+', choices=['mlp','ks2','ks3'], default=['mlp','ks2'])
    ap.add_argument('--ks3-backend', choices=['torch','fused'], default='fused')
    ap.add_argument('--ks3-rank', type=int, default=1)
    ap.add_argument('--ks3-subspaces', type=int, default=None,
                    help='Default: largest chart count within this port\'s full MLP FFN budget')
    ap.add_argument('--ks3-nonlinear', choices=['fp32','fp16','int8'], default='fp16')
    ap.add_argument('--ks3-compile', choices=['off','default','reduce-overhead'], default='off',
                    help='Compile KS3 FFNs only; reduce-overhead also enables GPU graph replay')
    ap.add_argument('--ks3-graph', action='store_true', help='Replay captured KS3 FFN training kernels')
    args = ap.parse_args()
    if min(args.steps,args.sequence,args.val_tokens,args.eval_every,args.threads) < 1 or args.max_seconds <= 30:
        ap.error('Positive budgets required; max-seconds must exceed 30')
    if args.ks3_rank < 1 or (args.ks3_subspaces is not None and args.ks3_subspaces < 1):
        ap.error('KS3 rank and subspace count must be positive')
    if len(args.models) != len(set(args.models)):
        ap.error('--models must not contain duplicates')
    if 'ks3' in args.models and args.ks3_backend == 'fused' and torch.device(args.device).type != 'cuda':
        ap.error('Fused KS3 needs --device cuda (or cuda:N)')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    train_tokens, val_tokens, manifest = data(args.data_dir)
    prefix = prefix_table().to(args.device)
    val = [torch.from_numpy(np.array(val_tokens[i:i+args.sequence+1], dtype=np.int64)).to(args.device)
           for i in range(0, args.val_tokens, args.sequence)]
    stream, _ = batches(train_tokens, 1, args.sequence, 777)
    probe = stream[0].to(args.device)
    pilots = {kind:pilot(kind,args,probe,prefix,val) for kind in args.models}
    safe = args.steps
    while safe > 0:
        evals = 1 + math.ceil(safe/args.eval_every)
        estimate = max(r['setup_seconds'] + safe*r['step_seconds'] + evals*r['validation_seconds'] for r in pilots.values())
        if estimate <= args.max_seconds*.65: break
        safe -= 1
    if safe < 20: raise RuntimeError('Fewer than 20 safe steps: reduce width/sequence/validation tokens')
    steps = safe if safe < 50 else safe//50*50
    config = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  fixed_steps=steps, tokens_per_step=args.sequence, precision='FP32 throughout; TF32 off',
                  upstream_commit=UPSTREAM, data_revision=DATA_REVISION, data_manifest=manifest,
                  pilots=pilots, torch=torch.__version__, numpy=np.__version__, python=platform.python_version(),
                  platform=platform.platform(), sources={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    if 'ks3' in args.models:
        config['ks3_precision'] = ('GS32 W8A8, FP16 quantization scales; FP32 masters/gradients; '
                                   + args.ks3_nonlinear + ' forward nonlinearities') if args.ks3_backend == 'fused' else 'Torch FP32 reference'
        if args.ks3_backend == 'fused':
            config['precision'] = 'Shared network/optimizer FP32; KS3 FFNs: ' + config['ks3_precision']
        for name in ('ks3.py','ks3_int8.py'):
            path=Path(__file__).parents[1]/'common'/name
            config['sources']['experiments/common/'+name]=sha(path)
    save(args.out/'config.json',config)
    print('FIXED_STEPS',steps,'TOKENS_PER_RUN',steps*args.sequence,flush=True)
    results=[]
    for seed in args.seeds:
        stream, offset=batches(train_tokens, steps, args.sequence,seed)
        stream=[t.to(args.device) for t in stream]
        for kind in args.models:
            results.append(train(kind,seed,args,steps,stream,prefix,val,dict(train=offset,validation=0)))
            save(args.out/'results.json',results)
    grouped={kind:[r for r in results if r['kind']==kind] for kind in args.models}
    summary=dict(steps=steps,tokens_per_run=steps*args.sequence,
                 note='Reduced-model comparison; negative paired deltas favor the compared FFN.')
    for kind,rows in grouped.items():
        summary[kind+'_mean_loss']=float(np.mean([r['final_val_loss'] for r in rows]))
        if kind!='mlp' and 'mlp' in grouped:
            controls={r['seed']:r for r in grouped['mlp']}
            summary['paired_'+kind+'_minus_mlp']=[r['final_val_loss']-controls[r['seed']]['final_val_loss'] for r in rows]
            summary[kind+'_ffn_ratio']=rows[0]['counts']['ffn']/grouped['mlp'][0]['counts']['ffn']
    if 'ks2_ffn_ratio' in summary:
        summary['ffn_ratio']=summary['ks2_ffn_ratio']  # Historical summary key.
    save(args.out/'summary.json',summary); print('SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__':main()
