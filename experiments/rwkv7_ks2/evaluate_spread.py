"""Four-offset full-shard evaluation for the completed projection experiment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import torch
from torch.nn import functional as F
import run
from gpu_recurrence import recurrence
from spread_projection import install
from sweep_plateau import environment


def worker(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    run.recurrence = recurrence
    root = Path(__file__).parent
    _, val, _ = run.prepare_data(root / 'gpu_runs/data', root / 'gpu_runs/data')
    batches = {}
    for offset in (0, 16, 32, 48):
        starts = torch.arange(offset, len(val) - 64, 64)
        windows = val[starts[:, None] + torch.arange(65)]
        count = len(windows)
        if (-count) % 64:
            windows = torch.cat((windows, windows[:(-count) % 64]))
        batches[offset] = (windows.reshape(-1, 64, 65).cuda(), count)
    rows = []
    for seed in args.seeds:
        path = args.runs / f'{args.arm}_{seed}'
        result = json.loads((path / 'results.json').read_text())
        cfg = result['arguments']
        model = install(run.Model(seed, cfg['variant'], width=cfg['width'], layers=cfg['layers'],
                                   ffn_charts=cfg['ffn_charts'], fused_cpu=False), result['coding']).cuda().eval()
        checkpoint = torch.load(path / 'checkpoint.pt', map_location='cuda', weights_only=True)
        assert checkpoint['step'] == cfg['steps'] == 15000
        model.load_state_dict(checkpoint['model'])
        del checkpoint
        static = batches[0][0][0].clone()
        with torch.no_grad():
            def per_window(batch):
                return F.cross_entropy(model(batch[:, :-1]).flatten(0, 1), batch[:, 1:].flatten(),
                                       reduction='none').view(-1, 64).mean(1)
            expected = torch.cat([per_window(chunk) for chunk in static.split(4)])
            actual = per_window(static)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
            batch_error = (actual - expected).abs().max().item()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                losses = per_window(static)
        offsets, all_losses = [], []
        for offset, (windows, count) in batches.items():
            collected = []
            for batch in windows:
                static.copy_(batch)
                graph.replay()
                collected.append(losses.clone())
            measured = torch.cat(collected)[:count]
            assert torch.isfinite(measured).all()
            all_losses.append(measured)
            offsets.append(dict(offset=offset, targets=count * 64, loss=measured.mean().item()))
        row = dict(arm=args.arm, seed=seed, loss=torch.cat(all_losses).mean().item(),
                   targets=sum(v['targets'] for v in offsets), by_offset=offsets,
                   batch4_vs_batch64_max_loss_error=batch_error)
        rows.append(row)
        print(json.dumps(row), flush=True)
        del graph, model
    destination = args.output or (args.runs / f'tiled_{args.arm}.json')
    destination.write_text(json.dumps(rows, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--arm', choices=['mlp', 'learned', 'fixed_input', 'fixed_both'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.arm:
        worker(args)
        return
    assert (args.runs / 'summary.json').exists(), 'Wait for all training to finish'
    arms = ['mlp', 'learned', 'fixed_input', 'fixed_both']

    def dispatch(item):
        gpu, arm = item
        cached = args.runs / f'tiled_{arm}.json'
        if cached.exists():
            rows = json.loads(cached.read_text())
            assert [r['seed'] for r in rows] == [11, 22, 33]
            assert all(r['arm'] == arm and r['targets'] == 1777024 for r in rows)
            return rows
        with (args.runs / f'tiled_{arm}.log').open('w') as log:
            subprocess.run([sys.executable, __file__, '--runs', str(args.runs), '--arm', arm],
                           env=environment(gpu), stdout=log, stderr=subprocess.STDOUT, check=True)
        return json.loads((args.runs / f'tiled_{arm}.json').read_text())

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = [row for group in pool.map(dispatch, enumerate(arms)) for row in group]
    result = dict(protocol='Same held-out shard tiled into 64-target windows at offsets 0/16/32/48; '
                           'nearly all bytes visited four times with different contexts, not independent test samples. '
                           'State resets each window; batch size 64 checked against 4, padding excluded.', rows=rows)
    Path(__file__).with_name('spread2_tiled_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
