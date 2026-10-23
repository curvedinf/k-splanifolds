"""Replay the saved nanoGPT MLP control scenario against fused KS3 on GPUs.

python -m experiments.nanogpt_ks2.compare_ks3 --data-dir /path/to/fineweb10B \
    --out experiments/nanogpt_ks2/runs/ks3-control --gpus 0 1 2 3

Uses the reference's exact step count, without the adaptive timing pilot cap.
Each GPU runs independent full models; this is experiment parallelism, not TP.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

from experiments.nanogpt_ks2 import run


def worker(cli):
    reference = json.loads((cli.reference / 'config.json').read_text())
    cfg = dict(reference['arguments'])
    cfg.update(out=cli.out, data_dir=cli.data_dir, device=f'cuda:{cli.gpus[0]}',
               max_seconds=3600, ks3_backend='fused', ks3_rank=1,
               ks3_subspaces=None, ks3_nonlinear='fp16', ks3_compile=cli.ks3_compile,
               ks3_graph=cli.ks3_graph)
    args = SimpleNamespace(**cfg)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(args.device)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    train, val, manifest = run.data(args.data_dir)
    for name, info in manifest.items():
        if info['sha256'] != reference['data_manifest'][name]['sha256']:
            raise ValueError(f'Reference data hash mismatch: {name}')
    prefix = run.prefix_table().to(args.device)
    validation = [torch.from_numpy(np.array(val[i:i+args.sequence+1], dtype=np.int64)).to(args.device)
                  for i in range(0, args.val_tokens, args.sequence)]
    steps = reference['fixed_steps']
    stream, offset = run.batches(train, steps, args.sequence, cli.seed)
    stream = [t.to(args.device) for t in stream]
    probe, _ = run.batches(train, 1, args.sequence, 777)
    # Compile/warm the actual training and validation paths before timing.
    pilot = run.pilot(cli.kind, args, probe[0].to(args.device), prefix, validation)
    paths = [Path(__file__), Path(run.__file__), Path(run.__file__).with_name('model.py'),
             Path(run.__file__).with_name('optim.py'),
             Path(__file__).parents[1]/'common/ks3.py',
             Path(__file__).parents[1]/'common/ks3_int8.py']
    run.save(args.out/'config.json', dict(
        arguments={k:str(v) if isinstance(v, Path) else v for k,v in cfg.items()},
        reference_config_sha256=run.sha(cli.reference/'config.json'),
        fixed_steps=steps, seed=cli.seed, kind=cli.kind, data_manifest=manifest,
        pilot=pilot, torch=torch.__version__, hip=torch.version.hip,
        python=platform.python_version(), gpu=torch.cuda.get_device_name(args.device),
        sources={str(p.relative_to(Path(__file__).resolve().parents[2])):run.sha(p) for p in paths},
        precision='Shared model and optimizer FP32; KS3 GS32 W8A8 / FP16 scales and nonlinearities / FP32 backward',
        timing='Synchronized full training steps; excludes warmup and validation. Wall time includes setup, validation, checkpoint save.',
        hardware_note='Historical control ran on CPU; GPU candidate sampling and floating-point kernels can differ. Both new arms use identical GPU sampling.'))
    result = run.train(cli.kind, cli.seed, args, steps, stream, prefix, validation,
                       dict(train=offset, validation=0))
    result['gpu'] = cli.gpus[0]
    result['step_ms'] = 1000*result['train_seconds']/steps
    run.save(args.out/'result.json', result)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    ap.add_argument('--ks3-compile', choices=['off','default','reduce-overhead'], default='off')
    ap.add_argument('--ks3-graph', action='store_true')
    ap.add_argument('--kind', choices=['mlp', 'ks3'], help=argparse.SUPPRESS)
    ap.add_argument('--seed', type=int, help=argparse.SUPPRESS)
    cli = ap.parse_args()
    if cli.kind:
        worker(cli)
        return
    cli.out.mkdir(parents=True, exist_ok=False)
    reference = json.loads((cli.reference/'config.json').read_text())
    jobs = [(kind, seed) for seed in reference['arguments']['seeds'] for kind in ('mlp', 'ks3')]
    def on_gpu(index, gpu):
        for kind, seed in jobs[index::len(cli.gpus)]:
            target = cli.out/f'{kind}_{seed}'
            command = [sys.executable, '-m', 'experiments.nanogpt_ks2.compare_ks3',
                       '--reference', str(cli.reference), '--data-dir', str(cli.data_dir),
                       '--out', str(target), '--gpus', str(gpu), '--kind', kind, '--seed', str(seed),
                       '--ks3-compile', cli.ks3_compile]
            if cli.ks3_graph:
                command.append('--ks3-graph')
            with (cli.out/f'{kind}_{seed}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            print('DONE', kind, seed, flush=True)
    with ThreadPoolExecutor(max_workers=len(cli.gpus)) as pool:
        futures = [pool.submit(on_gpu, i, gpu) for i,gpu in enumerate(cli.gpus)]
        for future in futures:
            future.result()
    results = [json.loads((cli.out/f'{kind}_{seed}'/'result.json').read_text()) for kind, seed in jobs]
    run.save(cli.out/'results.json', results)
    summary = dict(steps=reference['fixed_steps'], seeds=reference['arguments']['seeds'],
                   tokens_per_run=reference['fixed_steps']*reference['arguments']['sequence'])
    for kind in ('mlp','ks3'):
        rows = [r for r in results if r['kind']==kind]
        summary[kind] = {key:float(np.mean([r[key] for r in rows]))
                         for key in ('final_val_loss','step_ms','seconds','train_seconds')}
        summary[kind]['loss_sample_std'] = float(np.std([r['final_val_loss'] for r in rows], ddof=1))
        summary[kind]['ffn_params'] = rows[0]['counts']['ffn']
    summary['ks3_minus_mlp_loss'] = summary['ks3']['final_val_loss']-summary['mlp']['final_val_loss']
    summary['ks3_over_mlp_step_time'] = summary['ks3']['step_ms']/summary['mlp']['step_ms']
    historical = [r for r in json.loads((cli.reference/'results.json').read_text()) if r['kind']=='mlp']
    summary['historical_cpu_mlp'] = {key:float(np.mean([r[key] for r in historical]))
                                     for key in ('final_val_loss','seconds','train_seconds')}
    run.save(cli.out/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
