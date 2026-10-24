"""Descriptive activation spectra on fixed validation bytes, evaluated on CPU."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
import run
from study_models_v6 import build_model


def describe(x):
    x=x.double().flatten(0,-2)
    centered=x-x.mean(0)
    eigen=torch.linalg.eigvalsh(centered.T@centered/max(1,len(x)-1)).clamp_min(0)
    p=eigen/eigen.sum().clamp_min(1e-30)
    return dict(samples=len(x),features=x.shape[-1],rms=x.square().mean().sqrt().item(),
                abs_p99=torch.quantile(x.abs().flatten(),.99).item(),
                positive_fraction=(x>0).double().mean().item(),
                covariance_participation_rank=(eigen.sum().square()/eigen.square().sum().clamp_min(1e-30)).item(),
                covariance_entropy_rank=(-torch.where(p>0,p*p.clamp_min(1e-30).log(),0).sum()).exp().item(),
                top10_variance_fraction=p[-10:].sum().item())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    paths=['initial/history/fixed_both_11','wave2/dense31','wave2/dense31_relu2',
           'wave3/matched_init_mlp384','wave4_activation/matched_mlp368',
           'wave4_activation/spine_fused_11','wave4_activation/undamped_fused_11']
    rows=[]
    for relative in paths:
        path=a.root/relative;r=json.loads((path/'results.json').read_text())
        cfg=dict(r.get('study_config',dict(name='history')),fused_geometry=False,fused_partial=False)
        model=build_model(r['arguments']['seed'],cfg).eval()
        model.load_state_dict(torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)['model'])
        _,val,_=run.prepare_data(Path(r['arguments']['data']),path)
        positions=torch.randint(len(val)-65,(4,4),generator=torch.Generator().manual_seed(54321))
        batches=val[positions[:,:,None]+torch.arange(65)]
        record=[dict(coordinates=[],features=[],update=[]) for _ in model.blocks]
        hooks=[]
        for i,b in enumerate(model.blocks):
            cell=b.ffn
            def down_hook(module,args,out,i=i):record[i]['coordinates'].append(out.detach().reshape(-1,out.shape[-1]))
            def up_hook(module,args,out,i=i):
                record[i]['features'].append(args[0].detach().reshape(-1,args[0].shape[-1]))
                record[i]['update'].append(out.detach().reshape(-1,out.shape[-1]))
            hooks.append(cell.down.register_forward_hook(down_hook))
            hooks.append(cell.up.register_forward_hook(up_hook))
        with torch.no_grad():
            losses=[F.cross_entropy(model(b[:,:-1]).flatten(0,1),b[:,1:].flatten()).item() for b in batches]
        for h in hooks:h.remove()
        stats=[{name:describe(torch.cat(values)) for name,values in layer.items()} for layer in record]
        row=dict(run=relative,loss=sum(losses)/len(losses),layers=stats)
        rows.append(row);print(json.dumps(row),flush=True)
    output=dict(note='Descriptive, scale-dependent spectra on 1024 fixed validation targets, seed 54321. '
                'Participation rank is trace(C)^2/trace(C^2); entropy rank is exp(-sum p log p). '
                'Neither quantity establishes causal expressivity or a retraining advantage. '
                'History update includes the side stream; dense-model update is 96D.',rows=rows)
    (a.root/'feature_probe.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':main()
