"""Inspect trained KS2 term use and inference dependence on spine versus edges."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
import run
from study_models_v2 import build_model
from experiments.common.ks2_stabilized import _strength

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
 r=json.loads((a.run/'results.json').read_text());cfg=r.get('study_config',{'name':'history'});model=build_model(r['arguments']['seed'],dict(cfg,fused_geometry=False)).eval()
 model.load_state_dict(torch.load(a.run/'checkpoint.pt',map_location='cpu',weights_only=True)['model'])
 _,val,_=run.prepare_data(Path(r['arguments']['data']),a.run);pos=torch.randint(len(val)-65,(16,4),generator=torch.Generator().manual_seed(54321));batches=val[pos[:,:,None]+torch.arange(65)]
 records=[[] for _ in model.blocks];original=[]
 for i,b in enumerate(model.blocks):
  cell=b.ffn;original.append(cell.geo_step)
  def step(x,i=i,cell=cell):
   _,spine,atten,ex=cell.geometry_terms(x);edge=(atten*ex).sum(2)
   with torch.no_grad():records[i].append(dict(spine_rms=float(spine.square().mean().sqrt()),edge_rms=float(edge.square().mean().sqrt()),spine_edge_cosine=float((spine*edge).sum()/(spine.norm()*edge.norm()).clamp_min(1e-9))))
   return spine+edge
  cell.geo_step=step
 def evaluate():
  with torch.no_grad():return sum(float(F.cross_entropy(model(b[:,:-1]).flatten(0,1),b[:,1:].flatten())) for b in batches)/len(batches)
 base=evaluate();stats=[]
 for i,b in enumerate(model.blocks):
  cell=b.ffn;row={k:sum(r[k] for r in records[i])/len(records[i]) for k in records[i][0]}
  row.update(den_delta_mean=float(_strength(cell.geo_den_delta).mean()),den_weight_mean=float(_strength(cell.geo_den_weight).mean()),spine_coefficient_rms=float(cell.geo_spine.square().mean().sqrt()),t_edge_coefficient_rms=float(cell.geo_t_basis_edge.square().mean().sqrt()),x_edge_coefficient_rms=float(cell.geo_x_basis_edge.square().mean().sqrt()));stats.append(row);cell.geo_step=original[i]
 ablations={}
 for term in ('spine','edge'):
  for i,b in enumerate(model.blocks):
   cell=b.ffn
   def only(x,cell=cell,term=term):
    _,spine,atten,ex=cell.geometry_terms(x)
    return spine if term=='spine' else (atten*ex).sum(2)
   cell.geo_step=only;ablations[f'layer{i}_{term}_only']=evaluate();cell.geo_step=original[i]
 output=dict(base_loss=base,targets=4096,layers=stats,inference_ablations=ablations)
 (a.run/'geometry_terms_probe.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
if __name__=='__main__':main()
