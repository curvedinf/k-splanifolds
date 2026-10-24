import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from study_models_v14 import build_model,parameter_breakdown
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(591);tokens=torch.randint(256,(2,8));rows=[]
configs={j['config']['name']:j['config'] for j in json.loads((a.root/'wave12b_rank_freezing_manifest.json').read_text())['jobs']}
for name,config in configs.items():
 cfg=dict(config,fused_general=False);m=build_model(11,cfg);base=build_model(11,{k:v for k,v in cfg.items() if k!='frozen_geometry'})
 assert parameter_breakdown(m)['total']<=615936
 for b,o in zip(m.blocks,base.blocks):
  with torch.no_grad():b.ffn.up.weight.normal_(0,.02);o.ffn.up.weight.copy_(b.ffn.up.weight)
 torch.testing.assert_close(m(tokens),base(tokens),atol=0,rtol=0)
 F.cross_entropy(m(tokens).flatten(0,1),tokens.roll(-1,1).flatten()).backward()
 for key,p in m.named_parameters():
  if key=='blocks.0.attention.value_bias':continue
  assert p.grad is not None and torch.isfinite(p.grad).all(),key
 if cfg.get('frozen_geometry'):assert all('.geo_' not in n for n,p in m.named_parameters())
 row=dict(name=name,status='PASS',breakdown=parameter_breakdown(m),fixed_buffer_bytes=sum(b.numel()*b.element_size() for b in m.buffers()));rows.append(row);print(json.dumps(row),flush=True)
(a.root/'rank_freezing_cpu_check.json').write_text(json.dumps(dict(status='PASS',rows=rows),indent=2)+'\n')
