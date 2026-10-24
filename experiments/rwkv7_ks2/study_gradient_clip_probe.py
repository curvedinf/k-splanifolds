"""Read-only CPU gradient-group diagnostic on fixed training minibatches."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import torch
from torch.nn import functional as F
import run
from study_models_v9 import build_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    paths=['wave2/dense31_relu2','wave4_activation/spine_fused_11','wave4_activation/spine_frozen',
           'wave8_long/original_spine46_11','wave8_long/original_frozen46_11','wave8_long/original_undamped33_11']
    rows=[]
    for relative in paths:
        path=a.root/relative;r=json.loads((path/'completed.json').read_text())
        cfg=dict(r['study_config'],fused_geometry=False,fused_partial=False)
        model=build_model(r['arguments']['seed'],cfg)
        model.load_state_dict(torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)['model'])
        train,_,_=run.prepare_data(Path(r['arguments']['data']),path)
        positions=torch.randint(len(train)-65,(8,4),generator=torch.Generator().manual_seed(8675309))
        batches=train[positions[:,:,None]+torch.arange(65)];measurements=[]
        for batch in batches:
            model.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(batch[:,:-1]).flatten(0,1),batch[:,1:].flatten());loss.backward()
            squared={}
            for name,param in model.named_parameters():
                if param.grad is None:continue
                assert torch.isfinite(param.grad).all(),(relative,name)
                key='geometry' if '.geo_' in name else 'attention' if '.attention.' in name else 'input' if '.ffn.down.' in name else 'output' if '.ffn.up.' in name else 'other'
                squared[key]=squared.get(key,0.)+param.grad.double().square().sum().item()
            norm=sum(squared.values())**.5;geo=squared.get('geometry',0.)
            without=(sum(squared.values())-geo)**.5
            measurements.append(dict(loss=loss.item(),global_norm=norm,geometry_squared_norm_fraction=geo/(norm*norm),
                                     clip_factor=min(1.,1./(norm+1e-6)),clip_factor_excluding_geometry=min(1.,1./(without+1e-6)),squared_norm_by_group=squared))
        row=dict(run=relative,steps=r['arguments']['steps'],batches=measurements,
                 mean_geometry_squared_norm_fraction=statistics.mean(x['geometry_squared_norm_fraction'] for x in measurements),
                 mean_global_norm=statistics.mean(x['global_norm'] for x in measurements),
                 clipped_batches=sum(x['global_norm']>1 for x in measurements),
                 mean_clip_factor=statistics.mean(x['clip_factor'] for x in measurements),
                 mean_clip_factor_excluding_geometry=statistics.mean(x['clip_factor_excluding_geometry'] for x in measurements))
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='batches'}),flush=True)
    output=dict(protocol='CPU eager gradients; eight fixed B4 T64 training batches (2048 targets), seed8675309. No optimizer updates. Counterfactual norm exclusion changes only which gradient entries enter the global clipping norm. AdamW moment normalization means these factors are not learning-rate multipliers or measured parameter-step effects.',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows)
    (a.root/'gradient_clip_probe.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':main()
