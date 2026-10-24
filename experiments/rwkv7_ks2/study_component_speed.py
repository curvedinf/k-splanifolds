"""Isolated trained-block timing; not an additive full-training profiler."""
import argparse
import json
import random
import statistics
from pathlib import Path

import torch
import triton.testing
from torch.nn import functional as F

import run
from gpu_recurrence import recurrence
from study_fusion import with_fused_geometry
from study_models_v14 import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(31415)
    run.recurrence = recurrence
    paths = {
        'original_mlp': 'wave8_long/original_original_mlp_11',
        'history': 'wave8_long/original_history_11',
        'frozen_spine': 'wave8_long/original_frozen46_11',
        'learned_full31': 'wave2/dense31_relu2',
    }
    cases = []
    metadata = []

    def add_case(name, layer, component, function, inputs, parameters):
        values = tuple(x.detach().clone().requires_grad_(True) if torch.is_tensor(x) else x for x in inputs)
        targets = tuple(x for x in values if torch.is_tensor(x)) + tuple(p for p in parameters if p.requires_grad)
        initial = function(*values)
        outputs = initial if isinstance(initial, tuple) else (initial,)
        upstream = tuple(torch.randn_like(x) for x in outputs)

        def operation():
            output = function(*values)
            output = output if isinstance(output, tuple) else (output,)
            return torch.autograd.grad(output, targets, grad_outputs=upstream, allow_unused=True)

        gradients = operation()
        # RWKV layer zero declares value_bias but does not use value blending.
        expected_unused = 1 if component == 'attention' and layer == 0 else 0
        assert sum(x is None for x in gradients) == expected_unused, (name, layer, component)
        assert all(torch.isfinite(x).all() for x in gradients if x is not None), (name, layer, component)
        cases.append((name, layer, component, operation))

    for name, relative in paths.items():
        path = args.root / relative
        result = json.loads((path / 'results.json').read_text())
        config = dict(result['study_config'])
        if config['family'] == 'dense_ks2':
            config = with_fused_geometry(config)
        elif config['family'] == 'history':
            config['fused_geometry'] = True
        model = build_model(result['arguments']['seed'], config)
        model.load_state_dict(torch.load(path / 'checkpoint.pt', map_location='cpu', weights_only=True)['model'])
        model = model.cuda()
        captured = {}
        hooks = []

        def record(key):
            def hook(module, inputs):
                captured[key] = tuple(x.detach().clone() if torch.is_tensor(x) else x for x in inputs)
            return hook

        for layer, block in enumerate(model.blocks):
            hooks.append(block.attention.register_forward_pre_hook(record((layer, 'attention'))))
            if hasattr(block, 'ffn'):
                for key, module in [('ffn', block.ffn), ('down', block.ffn.down), ('up', block.ffn.up)]:
                    hooks.append(module.register_forward_pre_hook(record((layer, key))))
            else:
                hooks.append(block.key.register_forward_pre_hook(record((layer, 'down'))))
                hooks.append(block.value.register_forward_pre_hook(record((layer, 'up'))))
        train, _, _ = run.prepare_data(Path(result['arguments']['data']), path)
        positions = torch.randint(len(train) - 64, (4,), generator=torch.Generator().manual_seed(31415))
        tokens = train[positions[:, None] + torch.arange(64)].cuda()
        with torch.no_grad():
            model(tokens)
        for hook in hooks:
            hook.remove()
        for layer, block in enumerate(model.blocks):
            add_case(name, layer, 'attention', block.attention, captured[layer, 'attention'], block.attention.parameters())
            if hasattr(block, 'ffn'):
                cell = block.ffn
                add_case(name, layer, 'ffn', cell, captured[layer, 'ffn'], cell.parameters())
                add_case(name, layer, 'down', cell.down, captured[layer, 'down'], cell.down.parameters())
                add_case(name, layer, 'up', cell.up, captured[layer, 'up'], cell.up.parameters())
                with torch.no_grad():
                    coords = cell.down(*captured[layer, 'down']).reshape(-1, cell.num_subspaces, cell.subspace_rank)
                add_case(name, layer, 'geometry', cell.geo_step, (coords,), cell.parameters(recurse=False))
            else:
                def ffn(x, block=block):
                    return block.value(F.relu(block.key(x)).square())
                add_case(name, layer, 'ffn', ffn, captured[layer, 'down'], list(block.key.parameters()) + list(block.value.parameters()))
                add_case(name, layer, 'down', block.key, captured[layer, 'down'], block.key.parameters())
                add_case(name, layer, 'up', block.value, captured[layer, 'up'], block.value.parameters())
        metadata.append(dict(model=name, run=relative, config=config, steps=result['arguments']['steps']))
    measured = []
    rng = random.Random(31415)
    for repeat in range(3):
        order = cases.copy()
        rng.shuffle(order)
        for name, layer, component, operation in order:
            ms = triton.testing.do_bench_cudagraph(operation, rep=100)
            measured.append(dict(model=name, layer=layer, component=component, repeat=repeat, ms=ms))
        print(json.dumps(dict(repeat=repeat, cases=len(order), status='complete')), flush=True)
    summary = [dict(model=n, layer=l, component=c, median_ms=statistics.median(r['ms'] for r in measured if (r['model'], r['layer'], r['component']) == (n, l, c))) for n, l, c, _ in cases]
    args.out.write_text(json.dumps(dict(status='PASS', protocol='GPU0; three randomized captured timing rounds; trained real-input activations; B4 T64 FP32. Each case includes forward and input/learned-parameter derivatives with random fixed upstream gradients. Attention includes derivatives through its returned first-value tensor. Norms, FFN time mixing, residual additions, embedding/head, loss, clipping and optimizer are excluded. Isolated costs are not exactly additive and are not percentages of full-step time. FFN includes down, geometry/activation and up; do not add its subcomponents to it.', models=metadata, rows=measured, summary=summary), indent=2) + '\n')


if __name__ == '__main__':
    main()
