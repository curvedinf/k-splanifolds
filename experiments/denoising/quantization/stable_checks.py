from pathlib import Path
import json,torch
from stable_fp16 import make_model
R=Path(__file__).parent;torch.set_num_threads(8);rows=[]
m=make_model('stable',11);m.load_state_dict(torch.load(R/'stable_results/stable_11.pt',weights_only=False)['state_dict']);m.capture=True
for scale in [0.,1e-7,1e-5,.01,1.,10.,100.]:
 c=(torch.randn(128,32,8,generator=torch.Generator().manual_seed(123))*scale).requires_grad_();out=m.geo(c);out.square().mean().backward()
 rows.append(dict(scale=scale,output_finite=bool(torch.isfinite(out).all()),input_gradient_finite=bool(torch.isfinite(c.grad).all()),max_input_gradient=float(c.grad.abs().max()),stats=m.diagnostics));m.zero_grad(set_to_none=True)
(R/'stable_results/stress_checks.json').write_text(json.dumps(rows,indent=2));print([{k:v for k,v in r.items() if k!='stats'} for r in rows])
