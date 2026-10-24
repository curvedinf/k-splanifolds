"""CPU inference interventions; constants calibrated on training data only."""
import argparse,hashlib,json,types
from pathlib import Path
import torch
from torch.nn import functional as F
import run
from study_models_v9 import build_model


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 jobs=['wave3/dense_relu_seed33','wave3/spine46_relu','wave4_activation/spine_frozen','wave9_geometry_learning/spine_warps_only_11','wave8_long/original_spine46_11','wave8_long/original_frozen46_11','wave8_long/original_undamped33_11','wave8_long/larger_spine46_11','wave8_long/original_history_11']
 rows=[];dest=a.root/'geometry_counterfactual.json'
 for job in jobs:
  path=a.root/job;r=json.loads((path/'results.json').read_text());cfg=dict(r['study_config'],fused_geometry=False,fused_partial=False)
  model=build_model(r['arguments']['seed'],cfg).eval();model.load_state_dict(torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)['model'])
  train,val,_=run.prepare_data(Path(r['arguments']['data']),path)
  def batches(source,seed):
   pos=torch.randint(len(source)-65,(8,4),generator=torch.Generator().manual_seed(seed));return source[pos[:,:,None]+torch.arange(65)]
  train_batches=batches(train,47291);val_batches=batches(val,54321)
  cells=[b.ffn for b in model.blocks];original=[c.geo_step for c in cells];saved=[[] for c in cells]
  for i,c in enumerate(cells):
   def capture(x,i=i):
    g=original[i](x);saved[i].append((x.detach(),g.detach()));return g
   c.geo_step=capture
  with torch.no_grad():
   for batch in train_batches:model(batch[:,:-1])
  means=[];stats=[]
  for i,c in enumerate(cells):
   c.geo_step=original[i];x=torch.cat([z[0] for z in saved[i]]);g=torch.cat([z[1] for z in saved[i]]);mean=g.mean(0,keepdim=True);means.append(mean)
   t=4*torch.tanh(x.mean(-1,keepdim=True)/4);basis=c.spine_basis(t).double();gram=torch.einsum('nsm,nsj->smj',basis,basis)/len(basis)
   eigen=torch.linalg.eigvalsh(gram).clamp_min(0);condition=(eigen[:,-1]/eigen[:,0].clamp_min(1e-20)).sqrt()
   stats.append(dict(layer=i,coordinate_rms=x.square().mean().sqrt().item(),geometry_rms=g.square().mean().sqrt().item(),mean_fraction_squared_norm=(mean.square().sum()*len(g)/g.square().sum()).item(),basis_design_condition_median=condition.median().item(),basis_design_condition_max=condition.max().item(),basis_eigenvalues_median=eigen.median(0).values.tolist()))
  def evaluate():
   with torch.no_grad():return sum(F.cross_entropy(model(b[:,:-1]).flatten(0,1),b[:,1:].flatten()).item() for b in val_batches)/len(val_batches)
  base=evaluate();ablations=[]
  for mode in ('zero','mean_only','centered_only','half','one_and_half'):
   for layer in (-1,0,1,2,3):
    chosen=range(4) if layer==-1 else [layer]
    for i in chosen:
     def altered(x,i=i,mode=mode):
      if mode=='zero':return torch.zeros_like(x)
      if mode=='mean_only':return means[i].expand_as(x)
      g=original[i](x)
      if mode=='centered_only':return g-means[i]
      return g*(.5 if mode=='half' else 1.5)
     cells[i].geo_step=altered
    loss=evaluate();ablations.append(dict(mode=mode,layer=layer,loss=loss,delta=loss-base))
    for i in chosen:cells[i].geo_step=original[i]
  row=dict(job=job,baseline_loss=base,layers=stats,ablations=ablations);rows.append(row)
  output=dict(protocol='CPU eager inference; 2048 exploratory validation targets seed54321, means from 2048 training targets seed47291. No reserved holdout. Inference perturbations only, not retrained architecture comparisons. Layer -1 means all layers changed. Basis condition numbers computed on uncentered per-chart design matrices; singular directions clamped for numerical reporting.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows)
  dest.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(dict(job=job,base=base,all_layers=[x for x in ablations if x['layer']==-1])),flush=True)
if __name__=='__main__':main()
