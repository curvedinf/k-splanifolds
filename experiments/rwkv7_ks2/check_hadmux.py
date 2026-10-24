"""Check fixed routing, original controls, geometry gradients and GPU execution."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
import run
from side_stream import BaseModel
from side_history import HistoryModel
from hadmux_models import build_model, make_optimizer, parameter_breakdown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--stride', type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    spec = json.loads(args.manifest.read_text())
    configs = list({json.dumps(j['config'], sort_keys=True): j['config'] for j in spec['jobs']}.values())
    torch.manual_seed(7193)
    tokens = torch.randint(256, (2, 8))
    records = []
    if args.cpu:
        for seed in (11, 22, 33):
            original = BaseModel(seed, 'mlp', width=96, layers=4, fused_cpu=False)
            control = build_model(seed, dict(name='original_mlp', family='mlp'))
            assert all(torch.equal(v, control.state_dict()[k]) for k, v in original.state_dict().items())
            reference = HistoryModel(seed, 2048, 64)
            for cfg in configs:
                model = build_model(seed, dict(cfg, fused_general=False))
                with torch.no_grad():
                    torch.testing.assert_close(model(tokens), original(tokens), atol=0, rtol=0)
                features = cfg.get('features', 512)
                for i, block in enumerate(model.blocks):
                    for key, value in block.attention.state_dict().items():
                        assert torch.equal(value, original.blocks[i].attention.state_dict()[key])
                    assert block.main_width == 96 and block.side_width == cfg.get('side_width', 2048)
                    assert block.ffn.num_subspaces * block.ffn.subspace_rank == features
                    assert not any(p.requires_grad for n, p in block.ffn.down.named_parameters() if n == 'weight')
                    if cfg.get('side_width', 2048) == 2048 and features == 512:
                        assert torch.equal(block.ffn.down.weight, reference.blocks[i].ffn.down.weight)
                        assert torch.equal(block.ffn.up.matrix, reference.blocks[i].ffn.up.matrix)
                if cfg['name'] == 'had_full_r8':
                    assert all(torch.equal(v, reference.state_dict()[k]) for k, v in model.state_dict().items())
                records.append(dict(seed=seed, config=cfg['name'], status='PASS',
                                    learned_per_ffn=[sum(p.numel() for p in b.ffn.parameters()) for b in model.blocks],
                                    parameters=parameter_breakdown(model)))
                del model
            del original, control, reference
        result = dict(status='PASS', original_mlp_all_three_seeds_unchanged=True,
                      source_sha256=hashlib.sha256(Path(__file__).with_name('hadmux_models.py').read_bytes()).hexdigest(),
                      exact_projection_preservation=True, records=records)
        (args.manifest.parent/'hadmux_cpu_check.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(dict(status='PASS', cpu_cases=len(records))), flush=True)
        return
    from gpu_recurrence import recurrence
    run.recurrence = recurrence
    for cfg in configs[args.shard::args.stride]:
        eager = build_model(11, dict(cfg, fused_general=False)).blocks[-1].ffn.cuda()
        fused = build_model(11, cfg).blocks[-1].ffn.cuda()
        with torch.no_grad():
            eager.up.weight.normal_(0, .02)
            fused.up.weight.copy_(eager.up.weight)
        width = eager.down.weight.shape[1]
        x = torch.randn(2, 8, width, device='cuda', requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        left, right = eager(x), fused(y)
        torch.testing.assert_close(left, right, atol=3e-5, rtol=3e-5)
        weight = torch.randn_like(left)
        (left*weight).sum().backward()
        (right*weight).sum().backward()
        torch.testing.assert_close(x.grad, y.grad, atol=3e-5, rtol=3e-5)
        for (name, a), (other, b) in zip(eager.named_parameters(), fused.named_parameters()):
            assert name == other
            torch.testing.assert_close(a.grad, b.grad, atol=3e-5, rtol=3e-5,
                                       msg=lambda message: name+' '+message)
        del eager, fused, x, y, left, right, weight
        model = build_model(11, cfg).cuda()
        optimizer = make_optimizer(model)
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(tokens.cuda()).flatten(0, 1), tokens.roll(-1, 1).cuda().flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        for name, param in model.named_parameters():
            if name == 'blocks.0.attention.value_bias':
                assert param.grad is None
            else:
                assert param.grad is not None and torch.isfinite(param.grad).all(), name
        changed = tokens.clone()
        changed[:, 5:] = (changed[:, 5:] + 1) % 256
        with torch.no_grad():
            torch.testing.assert_close(model(tokens.cuda())[:, :5], model(changed.cuda())[:, :5], atol=2e-6, rtol=2e-6)
        print(json.dumps(dict(status='PASS', config=cfg['name'], parameters=parameter_breakdown(model))), flush=True)
        del model, optimizer


if __name__ == '__main__':
    main()
