"""CPU-only descriptive geometry trajectories on fixed exploratory validation bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
import run
from study_models_v9 import build_model


def rms(x):
    return x.double().square().mean().sqrt().item()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    dest=a.root/'geometry_trajectory.json'
    source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    existing=json.loads(dest.read_text()) if dest.exists() else {}
    previous=existing.get('rows',[]) if existing.get('source_sha256')==source_sha else []
    previous=[r for r in previous if 'spine_argument_negative_fraction' in r['layers'][0]]
    known={(r['job'],r['step']) for r in previous}
    rows=previous.copy()
    for path in sorted((a.root/'wave8_long').glob('*/completed.json')):
        result=json.loads(path.read_text())
        cfg=result['study_config']
        if cfg.get('geometry_terms') not in ('spine','undamped') or result['arguments']['seed']!=11:
            continue
        cfg=dict(cfg,fused_geometry=False,fused_partial=False)
        initial=build_model(11,cfg).state_dict()
        _,val,_=run.prepare_data(Path(result['arguments']['data']),path.parent)
        positions=torch.randint(len(val)-65,(4,4),generator=torch.Generator().manual_seed(54321))
        batches=val[positions[:,:,None]+torch.arange(65)]
        for step in (0,15000,50000,100000):
            if (path.parent.name,step) in known:continue
            model=build_model(11,cfg).eval()
            if step:
                checkpoint=path.parent/('checkpoint.pt' if step==100000 else f'checkpoint_{step}.pt')
                model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'])
            coords=[[] for _ in model.blocks]
            hooks=[]
            for i,b in enumerate(model.blocks):
                def capture(module,args,out,i=i):
                    coords[i].append(out.detach().reshape(-1,model.blocks[i].ffn.num_subspaces,8))
                hooks.append(b.ffn.down.register_forward_hook(capture))
            with torch.no_grad():
                losses=[F.cross_entropy(model(batch[:,:-1]).flatten(0,1),batch[:,1:].flatten()).item() for batch in batches]
            for hook in hooks:hook.remove()
            layers=[]
            with torch.no_grad():
                for i,b in enumerate(model.blocks):
                    cell=b.ffn
                    c=torch.cat(coords[i]);g=cell.geo_step(c);updated=c+g
                    t=4*torch.tanh(c.mean(-1,keepdim=True)/4)
                    warp=cell.warp(t,cell.geo_raw_scale,cell.geo_raw_shift)
                    scale=F.softplus(cell.geo_raw_scale)+1e-4
                    shift=.5*torch.tanh(cell.geo_raw_shift)
                    coefficients=cell.geo_spine
                    base=initial[f'blocks.{i}.ffn.geo_spine']
                    layers.append(dict(layer=i,coordinate_rms=rms(c),geometry_rms=rms(g),
                        spine_argument_mean=t.mean().item(),spine_argument_rms=rms(t),
                        spine_argument_negative_fraction=(t<0).float().mean().item(),
                        spine_argument_above_one_fraction=(t>1).float().mean().item(),
                        geometry_mean_rms=rms(g.mean(0)),geometry_centered_rms=rms(g-g.mean(0,keepdim=True)),
                        geometry_mean=g.mean().item(),geometry_positive_fraction=(g>0).float().mean().item(),
                        geometry_relative_rms=rms(g)/max(rms(c),1e-30),
                        coordinate_positive_fraction=(c>0).float().mean().item(),
                        updated_positive_fraction=(updated>0).float().mean().item(),
                        geometry_sign_flip_fraction=((c>0)!=(updated>0)).float().mean().item(),
                        feature_relative_change=rms(F.relu(updated).square()-F.relu(c).square())/max(rms(F.relu(c).square()),1e-30),
                        spine_warp_saturated_fraction=(warp.abs()>.95).float().mean().item(),
                        spine_warp_scale_mean=scale.mean().item(),spine_warp_scale_min=scale.min().item(),
                        spine_warp_scale_max=scale.max().item(),spine_warp_shift_rms=rms(shift),
                        spine_coefficient_rms=rms(coefficients),spine_coefficient_change_rms=rms(coefficients-base),
                        input_projection_rms=rms(cell.down.weight),output_projection_rms=rms(cell.up.weight)))
            row=dict(job=path.parent.name,step=step,sampled_loss=sum(losses)/len(losses),layers=layers)
            rows.append(row)
            print(json.dumps(dict(job=row['job'],step=step,sampled_loss=row['sampled_loss'])),flush=True)
            dest.write_text(json.dumps(dict(protocol='CPU eager inference; 1024 fixed exploratory validation targets, seed54321; no reserved holdout. Seed11 only. Descriptive, not a causal or retraining comparison. Parameter drift is from exact same initialization. Warp saturation means absolute tanh output above0.95.',source_sha256=source_sha,rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
