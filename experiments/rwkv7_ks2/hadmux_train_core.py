"""GPU follow-up: original equations, FP32, HIP graphs, paired fixed-step runs.

Compilation, graph setup, and evaluation are excluded from training_seconds.
Use --steps 0 to benchmark; benchmark/warmup updates are reset before training.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

import run as original
from gpu_recurrence import recurrence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--variant', choices=['mlp', 'ks2', 'ks2_ffn', 'inner_ks2_ffn', 'inner_mlp_plus_ks2'], required=True)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--ffn-charts', type=int, default=11)
    parser.add_argument('--control-update-scale', action='store_true')
    parser.add_argument('--steps', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--sequence-length', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    if args.control_update_scale and args.variant not in ('ks2_ffn', 'inner_ks2_ffn', 'inner_mlp_plus_ks2'):
        parser.error('Update scaling requires KS2 FFNs')
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=True)
    train, val, manifest = original.prepare_data(args.data, args.out)
    original.recurrence = recurrence
    model = original.Model(args.seed, args.variant, ffn_charts=args.ffn_charts,
                           width=args.width, layers=args.layers, fused_cpu=False).cuda()
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    # Preserve the original optimizer groups and AdamW shrinkage rule.
    base = original.make_optimizer(model, args.control_update_scale, args.ffn_charts)
    groups = []
    for group in base.param_groups:
        groups.append(dict(params=group['params'], weight_decay=group['weight_decay'],
                           lr=torch.tensor(group['lr'], device='cuda'),
                           lr_scale=group['lr_scale']))
    optimizer = torch.optim.AdamW(groups, capturable=True, foreach=True)
    del base
    count = max(args.steps, 1000)
    positions = torch.randint(len(train) - args.sequence_length - 1, (count, args.batch_size),
                              generator=torch.Generator().manual_seed(args.seed + 10000))
    val_positions = torch.randint(len(val) - args.sequence_length - 1, (32, args.batch_size),
                                  generator=torch.Generator().manual_seed(12345))
    offsets = torch.arange(args.sequence_length + 1)
    batches = train[positions[:, :, None] + offsets].cuda()
    val_batches = val[val_positions[:, :, None] + offsets].cuda()
    static_batch = batches[0].clone()
    invalid_steps = torch.zeros((), device='cuda', dtype=torch.int64)

    def update():
        optimizer.zero_grad(set_to_none=False)
        loss = F.cross_entropy(model(static_batch[:, :-1]).flatten(0, 1),
                               static_batch[:, 1:].flatten())
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=True)
        invalid_steps.add_((~(torch.isfinite(loss) & torch.isfinite(norm))).long())
        optimizer.step()
        return loss, norm

    def reset():
        model.load_state_dict(initial)
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    value.zero_()
        optimizer.zero_grad(set_to_none=False)
        invalid_steps.zero_()
        torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            update()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    reset()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_loss, static_norm = update()
    reset()
    # Include the same batch copies and LR writes used during actual training.
    begin = time.perf_counter()
    for step in range(100):
        static_batch.copy_(batches[step])
        for group in optimizer.param_groups:
            group['lr'].fill_(.001 * group['lr_scale'])
        graph.replay()
    torch.cuda.synchronize()
    milliseconds = (time.perf_counter() - begin) * 10
    reset()
    print(json.dumps(dict(benchmark_ms_per_step=milliseconds,
                          approx_120s_steps=int(120000 / milliseconds),
                          setup_seconds=time.perf_counter() - started)), flush=True)
    if not args.steps:
        result = dict(arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                      benchmark_ms_per_step=milliseconds)
        (args.out / 'benchmark.json').write_text(json.dumps(result, indent=2) + '\n')
        return

    eval_input = val_batches[0].clone()
    model.eval()
    with torch.no_grad():
        model(eval_input[:, :-1])
        eval_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(eval_graph):
            eval_loss = F.cross_entropy(model(eval_input[:, :-1]).flatten(0, 1),
                                        eval_input[:, 1:].flatten())
    model.train()

    def evaluate():
        losses = []
        for batch in val_batches:
            eval_input.copy_(batch)
            eval_graph.replay()
            losses.append(eval_loss.clone())
        return torch.stack(losses).mean().item()

    history = [dict(step=0, val_loss=evaluate())]
    print(json.dumps(history[-1]), flush=True)
    training_seconds = 0.
    # Report 1000-step checkpoints, plus finer early points for historical checks.
    checkpoints = {100, 200, 500, args.steps} | set(range(1000, args.steps + 1, 1000))
    torch.cuda.synchronize()
    segment_start = time.perf_counter()
    for step in range(1, args.steps + 1):
        static_batch.copy_(batches[step - 1])
        lr = .001 * (.2 + .8 * .5 * (1 + math.cos(math.pi * step / args.steps)))
        for group in optimizer.param_groups:
            group['lr'].fill_(lr * group['lr_scale'])
        graph.replay()
        if step in checkpoints:
            torch.cuda.synchronize()
            training_seconds += time.perf_counter() - segment_start
            if invalid_steps.item():
                raise RuntimeError('Nonfinite loss or gradient during graph replay')
            entry = dict(step=step, train_loss=static_loss.item(),
                         gradient_norm=static_norm.item(), val_loss=evaluate(),
                         training_seconds=training_seconds)
            history.append(entry)
            print(json.dumps(entry), flush=True)
            if (step == 15000 or step % 10000 == 0) and step < args.steps:
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step),args.out/f'checkpoint_{step}.pt')
            torch.cuda.synchronize()
            segment_start = time.perf_counter()
    sources = ['run.py', 'hadmux_train_core.py', 'gpu_recurrence.py',
               '../common/ks2.py', '../common/ks2_stabilized.py']
    hashes = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
              for name in sources}
    result = dict(arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  history=history, parameters=sum(p.numel() for p in model.parameters()),
                  ffn_parameters_by_layer=[sum(p.numel() for module in
                      (([block.ffn] if hasattr(block, 'ffn') else [block.key, block.value])
                       + ([block.ffn_extra] if hasattr(block, 'ffn_extra') else []))
                      for p in module.parameters()) for block in model.blocks],
                  ffn_types=['mlp+ks2' if hasattr(block, 'ffn_extra') else
                             ('ks2' if hasattr(block, 'ffn') else 'mlp') for block in model.blocks],
                  training_seconds=training_seconds, wall_seconds=time.perf_counter() - started,
                  benchmark_ms_per_step=milliseconds, invalid_steps=invalid_steps.item(),
                  data=manifest, torch=torch.__version__, hip=torch.version.hip,
                  device=torch.cuda.get_device_name(), source_hashes=hashes,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                  buffer_bytes=sum(b.numel()*b.element_size() for b in model.buffers()),
                  precision='FP32', batch_size=args.batch_size, sequence_length=args.sequence_length,
                  training_targets=args.steps * args.batch_size * args.sequence_length,
                  note='Independent GPU runs; no tensor parallelism. Warmup/benchmark updates reset.')
    (args.out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=args.steps),
               args.out / 'checkpoint.pt')
    print(json.dumps(dict(final_loss=history[-1]['val_loss'], training_seconds=training_seconds)), flush=True)


if __name__ == '__main__':
    main()
