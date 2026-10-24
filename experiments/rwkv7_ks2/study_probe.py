"""CPU diagnostics on fixed validation windows; inference ablations are not retraining."""
import argparse,json,types
from pathlib import Path
import torch
from torch.nn import functional as F
import run
from study_models_v2 import build_model

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
 r=json.loads((a.run/'results.json').read_text());cfg=r.get('study_config',{'name':'baseline'});model=build_model(r['arguments']['seed'],dict(cfg,fused_geometry=False)).eval()
 model.load_state_dict(torch.load(a.run/'checkpoint.pt',map_location='cpu',weights_only=True)['model'])
 _,val,_=run.prepare_data(Path(r['arguments']['data']),a.run);positions=torch.randint(len(val)-65,(16,4),generator=torch.Generator().manual_seed(54321));batches=val[positions[:,:,None]+torch.arange(65)]
 records=[[] for b in model.blocks];originals=[]
 for i,b in enumerate(model.blocks):
  cell=b.ffn;original=cell.geo_step;originals.append(original)
  def wrap(x,original=original,i=i,cell=cell):
   y=original(x)
   with torch.no_grad():
    w=cell.warp(x.unsqueeze(-1),cell.geo_edge_raw_scale,cell.geo_edge_raw_shift)
    records[i].append(dict(coords_rms=float(x.square().mean().sqrt()),geometry_rms=float(y.square().mean().sqrt()),
                          cosine=float((x*y).sum()/(x.norm()*y.norm()).clamp_min(1e-9)),edge_saturated_fraction=float((w.abs()>.95).float().mean()),
                          mean_coordinate_std=float(x.mean(-1).std())))
   return y
  cell.geo_step=wrap
 def evaluate():
  with torch.no_grad():return sum(float(F.cross_entropy(model(batch[:,:-1]).flatten(0,1),batch[:,1:].flatten())) for batch in batches)/len(batches)
 base=evaluate();stats=[{key:sum(row[key] for row in layer)/len(layer) for key in layer[0]} for layer in records]
 for b,original in zip(model.blocks,originals):b.ffn.geo_step=original
 ablations={}
 for i in range(4):
  model.blocks[i].ffn.geo_step=lambda x:torch.zeros_like(x)
  ablations[f'zero_geometry_layer_{i}']=evaluate()
  model.blocks[i].ffn.geo_step=originals[i]
 for i in range(4):
  cell=model.blocks[i].ffn;original=cell.forward
  def only_geo(x,cell=cell):
   coords=cell.down(x).float().reshape(-1,cell.num_subspaces,cell.subspace_rank).clamp(-65504,65504)
   return cell.up(cell.geo_step(coords).reshape(*x.shape[:-1],-1))
  cell.forward=only_geo;ablations[f'zero_coordinate_bypass_layer_{i}']=evaluate();cell.forward=original
 output=dict(base_loss=base,targets=batches.shape[0]*256,per_layer=stats,inference_ablations=ablations,note='Small fixed validation sample. Removing a trained component measures dependence, not whether retraining without it would hurt.')
 (a.run/'mechanism_probe.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
if __name__=='__main__':main()
