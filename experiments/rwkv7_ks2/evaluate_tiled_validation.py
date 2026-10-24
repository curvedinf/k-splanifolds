"""Compare nearly the entire held-out shard at four context-window offsets."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
import run
from gpu_recurrence import recurrence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--replay-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    root = Path(__file__).parent / 'gpu_runs'
    _, val, manifest = run.prepare_data(root / 'data', root / 'data')
    run.recurrence = recurrence
    batch_size = 64
    batches = {}
    for offset in (0, 16, 32, 48):
        starts = torch.arange(offset, len(val) - 64, 64)
        windows = val[starts[:, None] + torch.arange(65)]
        count = len(windows)
        padding = (-count) % batch_size
        if padding:
            windows = torch.cat((windows, windows[:padding]))
        batches[offset] = (windows.reshape(-1, batch_size, 65).cuda(), count)
    paths = ([root / 'additive4' / 'mlp_replay_11'] if args.replay_only else
             [root / 'hybrid4_plateau' / f'mlp_{args.seed}', root / 'additive4' / f'additive_{args.seed}'])
    rows = []
    for path in paths:
        result = json.loads((path / 'results.json').read_text())
        config = result['arguments']
        assert config['steps'] == 105000 and result['data'] == manifest
        model = run.Model(config['seed'], config['variant'], width=config['width'], layers=config['layers'],
                          ffn_charts=config['ffn_charts'], fused_cpu=False).cuda().eval()
        checkpoint = torch.load(path / 'checkpoint.pt', map_location='cuda', weights_only=True)
        assert checkpoint['step'] == 105000
        model.load_state_dict(checkpoint['model'])
        del checkpoint
        static = batches[0][0][0].clone()
        with torch.no_grad():
            model(static[:, :-1])
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                losses = F.cross_entropy(model(static[:, :-1]).flatten(0, 1),
                                         static[:, 1:].flatten(), reduction='none').view(batch_size, 64).mean(1)
        values = []
        all_losses = []
        for offset, (windows, count) in batches.items():
            collected = []
            for batch in windows:
                static.copy_(batch)
                graph.replay()
                collected.append(losses.clone())
            measured = torch.cat(collected)[:count]
            assert torch.isfinite(measured).all()
            all_losses.append(measured)
            values.append(dict(offset=offset, targets=count * 64, loss=measured.mean().item()))
        row = dict(run=path.name, steps=105000, by_offset=values,
                   targets=sum(v['targets'] for v in values), loss=torch.cat(all_losses).mean().item())
        rows.append(row)
        print(json.dumps(row), flush=True)
        del graph, model
    args.out.write_text(json.dumps(dict(
        protocol='Tile the held-out byte shard with 64-target windows, at offsets 0/16/32/48. State resets per window. Nearly all bytes evaluated four times with different contexts; overlapping visits are not independent samples. Same existing validation shard, not a separate test set. Last incomplete windows omitted. Batch size 64; padding excluded.',
        rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
