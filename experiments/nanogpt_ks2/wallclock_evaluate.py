"""Reload research checkpoints and evaluate a specified, common token interval."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.nanogpt_ks2 import run
from experiments.nanogpt_ks2.bench_forward import load_checkpoint


def load_research_checkpoint(path, device):
    metadata = json.loads(Path(path).with_name('config.json').read_text())
    banks = metadata['candidate'].get('banks')
    if banks is None:
        model, checkpoint = load_checkpoint(path, device)
    else:
        import math
        from experiments.nanogpt_ks2.model import GPT
        from experiments.nanogpt_ks2.wallclock_cells import MixedKS3
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        cfg = {k:v for k,v in checkpoint['model_config'].items() if k!='ks3_banks'}
        model = GPT(**cfg)
        model.ff = torch.nn.ModuleDict({key:MixedKS3(cfg['width'],banks) for key in model.ff})
        model.load_state_dict(checkpoint['model'])
        old = (16,16,48)
        steps = checkpoint['history'][-1]['step']
        for step in range(steps):
            window = run.schedule(step,steps)[0]
            new = (window[0],window[0],window[1])
            for yarn,a,b in zip((model.yarn,model.yarn_paired_head,model.yarn_wide),old,new):
                if a!=b:yarn.attn_scale *= 1+.2*math.log(b/a)
            old = new
        model = model.to(device).eval()
    if metadata['candidate'].get('window_attention', False):
        from experiments.nanogpt_ks2.wallclock_attention import install_window_attention
        install_window_attention(model)
    if metadata['candidate'].get('amp', False):
        from types import MethodType
        original = model.forward
        def forward(self, *args):
            with torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
                output = original(*args)
            return output.float()
        model.forward = MethodType(forward, model)
    basis = metadata['candidate'].get('basis', 'hermite')
    if basis != 'hermite':
        from experiments.nanogpt_ks2.wallclock_cells import change_basis
        # Install the basis-specific forward, then restore the already-converted
        # learned coefficients; change_basis also converts initial coefficients.
        for cell in model.ff.values():
            saved = cell.coefficients.detach().clone()
            change_basis(cell, basis)
            with torch.no_grad():
                cell.coefficients.copy_(saved)
    return model, checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--offset', type=int, required=True)
    ap.add_argument('--tokens', type=int, required=True)
    ap.add_argument('--gpu', type=int, default=3)
    ap.add_argument('--data-dir', type=Path,
                    default=Path('/home/curved/modded-nanogpt-ssinf/data/fineweb10B'))
    args = ap.parse_args()
    if args.offset < 0 or args.tokens <= 0 or args.tokens % 128:
        ap.error('Use a nonnegative offset and a positive multiple of 128 tokens')
    if args.out.exists():
        ap.error('Output already exists; retain previous evaluations')
    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    _, val, manifest = run.data(args.data_dir)
    if args.offset + args.tokens + 1 > len(val):
        ap.error('Validation interval exceeds the shard')
    device = f'cuda:{args.gpu}'
    batches = [torch.from_numpy(np.array(val[i:i+129], dtype=np.int64)).to(device)
               for i in range(args.offset, args.offset+args.tokens, 128)]
    results = []
    for path in args.checkpoints:
        recorded = json.loads(path.with_name('config.json').read_text())['data_manifest']
        if any(manifest[name]['sha256']!=info['sha256'] for name,info in recorded.items()):
            raise ValueError(f'Data hash mismatch for {path}')
        model, checkpoint = load_research_checkpoint(path, device)
        value = run.validate(model, batches)
        row = dict(checkpoint=str(path), checkpoint_sha256=run.sha(path),
                   loss=value, steps=checkpoint['history'][-1]['step'],
                   original_loss=checkpoint['history'][-1]['val_loss'])
        if args.offset==0 and args.tokens==checkpoint['args']['val_tokens']:
            row['reload_loss_difference']=value-row['original_loss']
            if abs(row['reload_loss_difference'])>1e-6:
                raise RuntimeError(f'Checkpoint reload did not reproduce the original loss: {row}')
        results.append(row)
        print(json.dumps(row), flush=True)
        del model, checkpoint
    run.save(args.out, dict(offset=args.offset, tokens=args.tokens, context=128,
                            data_manifest=manifest, results=results))


if __name__ == '__main__':
    main()
