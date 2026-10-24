"""Short fixed-coordinate RHAC Hadamard projection ablation; uses run_gpu protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
import run
import run_gpu
from spread_projection import install


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--coding', choices=['learned', 'fixed_input', 'fixed_both'], required=True)
    options, remaining = parser.parse_known_args()
    out = Path(remaining[remaining.index('--out') + 1])
    if (out / 'results.json').exists():
        raise RuntimeError('Use a fresh run directory')
    factory = run.Model

    def make_model(*args, **kwargs):
        return install(factory(*args, **kwargs), options.coding)

    run.Model = make_model
    sys.argv = [sys.argv[0], *remaining]
    run_gpu.main()
    path = out / 'results.json'
    if not path.exists():
        return
    result = json.loads(path.read_text())
    cfg = result['arguments']
    result['coding'] = options.coding
    result['coding_protocol'] = dict(
        width=cfg['width'], charts=cfg['ffn_charts'], rank=8,
        view_seed=4242 + 99173, hadamard_width=8,
        fold='signed orthonormal 8-way Hadamard groups; sum groups / sqrt(group count)',
        input_coverage='all 96 inputs in every coordinate of both charts',
        output='fixed transpose with unit row norms and 96 zero-init learned output gains'
        if options.coding == 'fixed_both' else 'zero-init learned dense output',
        note='Adaptation of RHAC SpreadViews, not a reproduction of depth-history conditioning; all input gradients retained.')
    for name in ('spread_projection.py', 'run_spread_gpu.py'):
        result['source_hashes'][name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    reference = Path(__file__).resolve().parents[3] / 'SDGraft/RHAC/src/rhac/rhac.py'
    result['rhac_reference'] = dict(path=str(reference), sha256=hashlib.sha256(reference.read_bytes()).hexdigest())
    train, val, _ = run.prepare_data(Path(cfg['data']), out)
    model = make_model(cfg['seed'], cfg['variant'], width=cfg['width'], layers=cfg['layers'],
                       ffn_charts=cfg['ffn_charts'], fused_cpu=False).cuda().eval()
    checkpoint = torch.load(out / 'checkpoint.pt', map_location='cuda', weights_only=True)
    model.load_state_dict(checkpoint['model'])
    assert all(int(s['step'].item()) == cfg['steps'] for s in checkpoint['optimizer']['state'].values())
    assert all(torch.isfinite(p).all() for p in model.parameters())
    del checkpoint
    # Graph evaluation with same batch shape and samples as preceding studies.
    static = torch.zeros(4, 65, dtype=torch.long, device='cuda')
    with torch.no_grad():
        model(static[:, :-1])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loss = F.cross_entropy(model(static[:, :-1]).flatten(0, 1), static[:, 1:].flatten())
    result['evaluation'] = {}
    for name, source, seed, count in [('original_val', val, 12345, 32),
                                      ('expanded_val', val, 54321, 256),
                                      ('sampled_train', train, 98765, 256)]:
        positions = torch.randint(len(source) - 65, (count, 4),
                                  generator=torch.Generator().manual_seed(seed))
        batches = source[positions[:, :, None] + torch.arange(65)].cuda()
        losses = []
        for batch in batches:
            static.copy_(batch)
            graph.replay()
            losses.append(loss.clone())
        result['evaluation'][name] = torch.stack(losses).mean().item()
    assert abs(result['evaluation']['original_val'] - result['history'][-1]['val_loss']) < 2e-6
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(evaluation=result['evaluation'], coding=options.coding)), flush=True)


if __name__ == '__main__':
    main()
