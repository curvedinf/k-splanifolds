import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from study_models_v12 import build_model,parameter_breakdown
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(519);tokens=torch.randint(256,(2,8));rows=[]
for job in json.loads(a.manifest.read_text())['jobs'][:6]:
 cfg=dict(job['config'],fused_partial=False,fused_geometry=False);m=build_model(11,cfg)
 base=build_model(11,{k:v for k,v in cfg.items() if k!='geometry_calibration'})
 assert parameter_breakdown(m)['total']<=615936
 for b,o in zip(m.blocks,base.blocks):
  with torch.no_grad():b.ffn.up.weight.normal_(0,.02);o.ffn.up.weight.copy_(b.ffn.up.weight)
 torch.testing.assert_close(m(tokens),base(tokens),atol=0,rtol=0)
 F.cross_entropy(m(tokens).flatten(0,1),tokens.roll(-1,1).flatten()).backward()
 for name,p in m.named_parameters():
  if name=='blocks.0.attention.value_bias':continue
  assert p.grad is not None and torch.isfinite(p.grad).all(),name
 if cfg.get('geometry_trainable')=='warps':
  assert all('raw_scale' in n or 'raw_shift' in n or 'feature_bias' in n for n,p in m.named_parameters() if '.geo_' in n)
 row=dict(name=cfg['name'],status='PASS',parameters=parameter_breakdown(m));rows.append(row);print(json.dumps(row),flush=True)
a.out.write_text(json.dumps(dict(status='PASS',rows=rows),indent=2)+'\n')
