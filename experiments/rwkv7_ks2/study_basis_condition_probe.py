"""Compare equivalent basis design conditioning on identical trained coordinates."""
import argparse,json,hashlib
from pathlib import Path
import torch
import run
from study_models_v9 import build_model
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(2);torch.set_num_interop_threads(1);rows=[]
for job in ['wave3/dense_relu_seed33','wave4_activation/spine_fused_11','wave8_long/original_spine46_11','wave8_long/original_frozen46_11']:
 path=a.root/job;r=json.loads((path/'results.json').read_text());cfg=dict(r['study_config'],fused_geometry=False,fused_partial=False)
 m=build_model(r['arguments']['seed'],cfg).eval();m.load_state_dict(torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)['model'])
 train,_,_=run.prepare_data(Path(r['arguments']['data']),path);pos=torch.randint(len(train)-65,(4,4),generator=torch.Generator().manual_seed(7124));batches=train[pos[:,:,None]+torch.arange(65)]
 coords=[[] for _ in m.blocks];handles=[]
 for i,b in enumerate(m.blocks):
  def capture(module,args,out,i=i):coords[i].append(out.detach().reshape(-1,m.blocks[i].ffn.num_subspaces,8))
  handles.append(b.ffn.down.register_forward_hook(capture))
 with torch.no_grad():
  for batch in batches:m(batch[:,:-1])
 for handle in handles:handle.remove()
 for i,b in enumerate(m.blocks):
  cell=b.ffn;c=torch.cat(coords[i]);t=4*torch.tanh(c.mean(-1,keepdim=True)/4);z=cell.warp(t,cell.geo_raw_scale,cell.geo_raw_shift)
  for kind in ('hermite','powers','taylor'):
   if kind=='hermite':basis=cell.spine_basis(t)
   else:
    d2,d3=(2.,6.) if kind=='taylor' else (1.,1.)
    basis=torch.cat([torch.ones_like(t),t,t.square()/d2,t.pow(3)/d3,2*z.square()-1,4*z.pow(3)-3*z],-1)
   with torch.no_grad():
    design=basis.transpose(0,1).double();singular=torch.linalg.svdvals(design)
    condition=singular[:,0]/singular[:,-1].clamp_min(1e-20)
    feature_rms=design.square().mean(1).sqrt();balanced=design/feature_rms[:,None].clamp_min(1e-20)
    balanced_singular=torch.linalg.svdvals(balanced);balanced_condition=balanced_singular[:,0]/balanced_singular[:,-1].clamp_min(1e-20)
   rows.append(dict(job=job,layer=i,basis=kind,condition_median=condition.median().item(),condition_max=condition.max().item(),rms_balanced_condition_median=balanced_condition.median().item(),basis_rms_median=feature_rms.median(0).values.tolist()))
 print(json.dumps(dict(job=job,layers=[{r['basis']:r['condition_median'] for r in rows if r['job']==job and r['layer']==i} for i in range(4)])),flush=True)
(a.root/'basis_condition_probe.json').write_text(json.dumps(dict(protocol='CPU inference on1024 fixed training targets,seed7124; uncentered per-chart1024x6 design SVD, float64. Same sampled coordinates for every equivalent basis; descriptive rather than an optimizer Hessian.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows),indent=2)+'\n')
