"""Measure first-update scale and gradient gating without full retraining."""
import argparse
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from run import Model, prepare_data, make_optimizer, set_learning_rate

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', type=Path, required=True)
data_root = parser.parse_args().data
train, _, _ = prepare_data(data_root, data_root)
starts = torch.randint(len(train)-65,(1000,4),generator=torch.Generator().manual_seed(10011))[0]
batch = train[starts[:,None]+torch.arange(65)]
rows = []
for width in (11,22,44,88,176):
    model = Model(11,'ks2_ffn',ffn_charts=width)
    optimizer = make_optimizer(model, True, width)
    cell = model.blocks[0].ffn
    captured = []
    hook = cell.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
    loss = F.cross_entropy(model(batch[:,:-1]).flatten(0,1),batch[:,1:].flatten())
    hook.remove()
    loss.backward()
    grad = lambda p:float(p.grad.norm()) if p.grad is not None else None
    row = dict(subspaces=width,initial_loss=float(loss.detach()),down_grad_norm=grad(cell.down.weight),spine_grad_norm=grad(cell.geo_spine),up_grad_norm=grad(cell.up.weight))
    row['whole_model_grad_norm_before_clip']=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
    set_learning_rate(optimizer, 1, 1000)
    optimizer.step()
    with torch.no_grad():
        inputs=captured[0]
        coords=cell.down(inputs).reshape(-1,width,8)
        geometry=cell.geo_step(coords)
        linear=F.linear(coords.flatten(1),cell.up.weight)
        nonlinear=F.linear(geometry.flatten(1),cell.up.weight)
        rms=lambda x:float(x.square().mean().sqrt())
        row.update(first_ffn_output_rms_after_one_update=rms(linear+nonlinear),linear_part_rms=rms(linear),geometry_part_rms=rms(nonlinear),projected_coordinate_rms=rms(coords),geometry_coordinate_rms=rms(geometry))
    rows.append(row)
    print(json.dumps(row),flush=True)
Path(__file__).with_name('controlled_width_diagnostics.json').write_text(json.dumps(rows,indent=2))
