"""Re-evaluate saved GPU runs on larger, fixed train/validation samples."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
import run
from gpu_recurrence import recurrence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ablate-added-ks2', action='store_true',
                        help='Also evaluate additive models with only their added KS2 outputs zeroed')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    train, val, _ = run.prepare_data(args.data, args.data)
    run.recurrence = recurrence
    batches = {}
    for name, source, seed, count in [('original_val', val, 12345, 32),
                                       ('expanded_val', val, 54321, 256),
                                       ('sampled_train', train, 98765, 256)]:
        positions = torch.randint(len(source) - 65, (count, 4),
                                  generator=torch.Generator().manual_seed(seed))
        batches[name] = source[positions[:, :, None] + torch.arange(65)].cuda()
    rows = []
    for path in sorted(args.runs.glob('*/results.json')):
        result = json.loads(path.read_text())
        config = result['arguments']
        model = run.Model(config['seed'], config['variant'],
                          ffn_charts=config['ffn_charts'], width=config.get('width', 128),
                          layers=config.get('layers', 2), fused_cpu=False).cuda().eval()
        checkpoint = torch.load(path.with_name('checkpoint.pt'), map_location='cuda', weights_only=True)
        model.load_state_dict(checkpoint['model'])
        optimizer_steps = torch.stack([state['step'] for state in checkpoint['optimizer']['state'].values()])
        assert optimizer_steps.unique().tolist() == [float(config['steps'])]
        del checkpoint
        static_input = batches['original_val'][0].clone()
        with torch.no_grad():
            model(static_input[:, :-1])
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                loss = F.cross_entropy(model(static_input[:, :-1]).flatten(0, 1),
                                       static_input[:, 1:].flatten())
        row = dict(run=path.parent.name)
        for name, split in batches.items():
            losses = []
            for batch in split:
                static_input.copy_(batch)
                graph.replay()
                losses.append(loss.clone())
            row[name] = torch.stack(losses).mean().item()
        assert abs(row['original_val'] - result['history'][-1]['val_loss']) < 2e-6
        added = [block.ffn_extra.up.weight for block in model.blocks if hasattr(block, 'ffn_extra')]
        if args.ablate_added_ks2 and added:
            saved = [p.detach().clone() for p in added]
            with torch.no_grad():
                for p in added:
                    p.zero_()
            disabled = {}
            for name, split in batches.items():
                losses = []
                for batch in split:
                    static_input.copy_(batch)
                    graph.replay()
                    losses.append(loss.clone())
                disabled[name] = torch.stack(losses).mean().item()
            row['added_ks2_disabled_at_evaluation'] = disabled
            with torch.no_grad():
                for p, value in zip(added, saved):
                    p.copy_(value)
            restored = []
            for batch in batches['original_val']:
                static_input.copy_(batch)
                graph.replay()
                restored.append(loss.clone())
            assert abs(torch.stack(restored).mean().item() - row['original_val']) < 2e-6
        rows.append(row)
        print(json.dumps(row), flush=True)
        del graph, model
    args.out.write_text(json.dumps(dict(
        protocol='Fixed 256 batches x 4 sequences x 64 bytes per expanded split; same original shards, overlapping windows possible. Not an independent test set.',
        seeds=dict(expanded_val=54321, sampled_train=98765),
        added_branch_ablation=('Post-training inference-only intervention: zero added KS2 output projections, retain all learned MLP weights, then restore. Not a retrained MLP control.'
                               if args.ablate_added_ks2 else None), rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
