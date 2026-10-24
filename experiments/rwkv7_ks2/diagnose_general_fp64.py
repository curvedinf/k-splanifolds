"""Independent CPU float64 autograd reference for saved GPU stress failures."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
import run
from experiments.common.ks2_stabilized import _STRENGTH_OFFSET


def evaluate(x, p, basis, terms):
    x = x.clamp(-65504, 65504)
    t = 4 * torch.tanh(x.mean(-1, keepdim=True) / 4)
    def warp(v, scale, shift):
        shape = (1, -1) + (1,) * (v.ndim - 2)
        return torch.tanh((F.softplus(scale).view(shape) + 1e-4) * (v + .5 * shift.tanh().view(shape)))
    z = warp(t, p['geo_raw_scale'], p['geo_raw_shift'])
    if basis == 'hermite':
        b = (2*t**3-3*t**2+1, -2*t**3+3*t**2, t**3-2*t**2+t, t**3-t**2)
    elif basis == 'powers':
        b = (torch.ones_like(t), t, t**2, t**3)
    else:
        assert basis == 'taylor'
        b = (torch.ones_like(t), t, t**2/2, t**3/6)
    b = torch.cat((*b, 2*z**2-1, 4*z**3-3*z), -1)
    spine = torch.einsum('nsm,smr->nsr', b, p['geo_spine'])
    if terms == 'spine':
        return spine
    w = warp(x.unsqueeze(-1), p['geo_edge_raw_scale'], p['geo_edge_raw_shift'])
    edge_basis = torch.cat((torch.ones_like(w), w, 2*w**2-1, 4*w**3-3*w), -1)
    et = torch.einsum('nsm,smro->nsro', b, p['geo_t_basis_edge'])
    ex = torch.einsum('nsrk,skro->nsro', edge_basis, p['geo_x_basis_edge'])
    energy = x.square()
    total = energy.sum(-1, keepdim=True)
    deviation = (x - x.mean(-1, keepdim=True)).square()
    distance = deviation / (deviation + total / x.shape[-1]).clamp_min(1e-30)
    importance = energy / total.clamp_min(1e-30)
    da = 8 * torch.sigmoid(p['geo_den_delta'] + _STRENGTH_OFFSET)
    db = 8 * torch.sigmoid(p['geo_den_weight'] + _STRENGTH_OFFSET)
    denominator = 1 + da[None] * distance[..., None] + db[None] * importance[..., None]
    return spine + (et / denominator * ex).sum(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows = []
    for path in sorted(args.directory.glob('*.pt')):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        info = saved['info']
        x = saved['x'].double().requires_grad_()
        params = {n: p.double().requires_grad_(n in saved['parameter_names']) for n,p in saved['state'].items()}
        result = evaluate(x, params, info['basis'], info['terms'])
        expected = torch.autograd.grad(result, (x, *(params[n] for n in saved['parameter_names'])), grad_outputs=saved['upstream'].double())
        comparisons = []
        for index, (fp64, eager, fused) in enumerate(zip(expected, saved['expected'], saved['actual'])):
            eager, fused = eager.double(), fused.double()
            scale = max(fp64.norm().item(), 1e-30)
            worst = (eager-fused).abs().flatten().argmax().item()
            comparisons.append(dict(index=index, fp64_norm=fp64.norm().item(), eager_relative_l2=(eager-fp64).norm().item()/scale, fused_relative_l2=(fused-fp64).norm().item()/scale, eager_max_absolute=(eager-fp64).abs().max().item(), fused_max_absolute=(fused-fp64).abs().max().item(), worst_pairwise_component=dict(flat_index=worst, fp64=fp64.flatten()[worst].item(), eager=eager.flatten()[worst].item(), fused=fused.flatten()[worst].item())))
        forward=None
        if 'forward_expected' in saved:
            target=result.detach();eager=saved['forward_expected'].double();fused=saved['forward_actual'].double();scale=max(target.norm().item(),1e-30)
            forward=dict(eager_relative_l2=(eager-target).norm().item()/scale,fused_relative_l2=(fused-target).norm().item()/scale,eager_max_absolute=(eager-target).abs().max().item(),fused_max_absolute=(fused-target).abs().max().item())
        rows.append(dict(case=info, comparisons=comparisons,forward=forward))
    output = dict(status='DIAGNOSTIC', cases=len(rows), rows=rows, note='CPU float64 eager-autograd evaluation of the same saved FP32 inputs and parameters. Does not itself authorize any GPU kernel or change tolerances.')
    (args.directory/'fp64_comparison.json').write_text(json.dumps(output, indent=2)+'\n')
    for row in rows:
        print(json.dumps(dict(case=row['case']['label'], comparison=row['forward'] if row['case']['gradient_index'] is None else row['comparisons'][row['case']['gradient_index']]), indent=2))


if __name__ == '__main__':
    main()
