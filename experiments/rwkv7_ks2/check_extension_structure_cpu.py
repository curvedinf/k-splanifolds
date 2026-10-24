"""Verify extended factories preserve the original control and fused state layout."""
import argparse,hashlib,json
from pathlib import Path
import torch
import run
from study_models_v13 import build_model,make_optimizer,parameter_breakdown
from study_models_v10 import build_model as build_v10
from experiments.common.ks2_stabilized import KS2StabilizedCell
from ks2_general_gpu import NAMES
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);rows=[]
for seed in (11,22,33):
 reference=run.Model(seed,'mlp',width=96,layers=4,fused_cpu=False);model=build_model(seed,dict(name='original_mlp',family='mlp'))
 assert reference.state_dict().keys()==model.state_dict().keys()
 for name,tensor in reference.state_dict().items():assert torch.equal(tensor,model.state_dict()[name]),name
 first=run.make_optimizer(reference);second=make_optimizer(model)
 assert len(first.param_groups)==len(second.param_groups)
 for x,y in zip(first.param_groups,second.param_groups):
  assert {k:v for k,v in x.items() if k!='params'}=={k:v for k,v in y.items() if k!='params'}
  assert len(x['params'])==len(y['params'])
  for xp,yp in zip(x['params'],y['params']):assert torch.equal(xp,yp)
 rows.append(dict(seed=seed,status='PASS',parameters=parameter_breakdown(model)['total'],check='Exact original MLP state and optimizer groups'))
configs={j['config']['name']:j['config'] for j in json.loads((a.root/'wave11_conditioning_manifest.json').read_text())['jobs']}
eligible=[]
for name,cfg in configs.items():
 if cfg.get('geometry_input_norm') or cfg.get('geometry_placement'):continue
 old=build_v10(11,cfg);new=build_model(11,dict(cfg,fused_general=True))
 assert old.state_dict().keys()==new.state_dict().keys()
 for key,value in old.state_dict().items():assert torch.equal(value,new.state_dict()[key]),(name,key)
 eligible.append(dict(name=name,parameters=parameter_breakdown(new)['total'],status='EXACT_STATE'))
for rank in (1,2,4,8,16):
 cell=KS2StabilizedCell(5,5,3,rank,1);sizes=(6*rank,6*rank*rank,4*rank*rank,rank*rank,rank*rank,1,1,1,1)
 packed=torch.cat([getattr(cell,n).detach().reshape(3,-1) for n in NAMES],1)
 assert packed.shape==(3,6*rank+12*rank*rank+4)
 for value,name in zip(packed.split(sizes,1),NAMES):assert torch.equal(value.reshape_as(getattr(cell,name)),getattr(cell,name))
result=dict(status='PASS',original_control=rows,fused_factory_state=eligible,packed_ranks=[1,2,4,8,16],note='CPU structure/layout only. Actual fused numerical forward/backward/capture checks are still required on GPU.',source_sha256={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ['study_models_v13.py','ks2_general_gpu.py']})
(a.root/'extension_structure_cpu_check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status='PASS',original_seeds=3,fused_configs=len(eligible),ranks=5)))
